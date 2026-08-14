# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PyTorch reimplementation/fork of [HIL-SERL](https://github.com/rail-berkeley/hil-serl): distributed
SAC for robot manipulation, with human-in-the-loop interventions. Runs on MuJoCo sim, ManiSkill/SAPIEN
sim, and a real Franka FR3. The upstream project is JAX; this one is not — there is no Flax/Optax code.

Compared to upstream HIL-SERL: no reward classifier, and no BC or gradient-penalty regularization on
the *online* objective. There is an optional offline BC warm start (`--pretrain_steps`), but it runs
before online training and does not change the SAC loss.

## Commands

Dependencies are managed by `uv` (Python pinned to exactly 3.10.19). Prefix everything with `uv run`.

```bash
uv sync                       # base install
uv sync --extra maniskill     # + ManiSkill; a later bare `uv sync` REMOVES it, so keep the flag
```

Training is always **two processes**. Start the learner first:

```bash
# terminal 1
uv run python -m train.train_serl --exp_name=insert_sim \
  --demo_path=demo_data/<file>.pkl --checkpoint_path=checkpoints/insert_sim --learner
# terminal 2 (add --ip=<learner-ip> if remote)
uv run python -m train.train_serl --exp_name=insert_sim \
  --checkpoint_path=checkpoints/insert_sim --actor
```

Evaluation is the actor with `--eval_n_trajs=N` and no learner; it writes one MP4 per trajectory to
`<checkpoint_path>/videos`. `--eval_video_main_camera` defaults to `front`, which **does not exist in
the ManiSkill scene** — pass `render_camera` there or evaluation crashes. When evaluating a HIL task,
also pass `--config_override=hil.enabled=false`, or the numbers measure the expert.

Other entrypoints:

```bash
uv run python -m train.record_demo --exp_name=<task> --successes_needed=N   # collect demos
uv run python -m train.verify_demo --demo_path=demo_data/<file>.pkl         # check + replay a demo file
uv run python -m train.test_intervention --exp_name=<task>_human           # try teleop, records nothing
uv run python train/convert_lerobot_demo.py --dataset_path=... --output_path=...  # LeRobot -> .pkl
python -m infra.hardware.robot.franka_server --robot_ip=172.16.0.2 --robot_type=fr3  # real robot
```

`verify_demo` infers the task from the file name (`--exp_name` overrides), prints a pass/fail report,
and exits non-zero on failure so it works as a gate in a script. `--no_render` keeps the checks and
drops the cv2 window; it degrades to that automatically with no display. It replays the **stored
frames** rather than re-driving the simulator, and that is forced rather than chosen: `record_demo`'s
per-episode `env.reset()` takes no seed, so object placement is re-randomised and never recorded.
Feeding the stored actions into a fresh env would show the arm executing a good demo against a scene
whose carton is somewhere else — a silent failure that reads as corrupt data.

Two wrapper scripts cover the full flow and are the fastest way to smoke-test a change:

```bash
./scripts/run_hil_serl.sh insert_sim all        # demos -> short learner+actor -> eval
./scripts/run_insert_maniskill.sh prepare       # download HF dataset + convert demos
```

`run_hil_serl.sh` takes env-var overrides instead of file edits: `EXPERT`, `TRIGGER`, `THRESHOLD`,
`HUD`, `DEMO_TRIGGER`, `DEBUG`, `SEED`, `CHECKPOINT_PATH`, `DEMO_PATH`, and the smoke-run budgets
`SMOKE_DEMOS` / `SMOKE_STEPS` / `SMOKE_EVAL`. Both scripts forward extra args to `train_serl`, so
`./scripts/run_hil_serl.sh insert_sim learner --config_override=training.batch_size=128` works.

`train_serl` **always** initializes wandb, and wandb aborts the process outright when no API key is
configured — the learner dies before its first step while the actor silently retries a connection that
never comes. Pass `--debug` (wandb disabled) when there is no key; `run_hil_serl.sh` does this
automatically unless `WANDB_API_KEY` or `~/.netrc` exists.

**There is no test suite and no linter configured.** `infra/hardware/spacemouse/test_spacemouse.py` is
a manual hardware probe, not a pytest test. Verification means running a task end to end — usually
`run_hil_serl.sh <task> all` with small `SMOKE_DEMOS`/`SMOKE_STEPS`/`SMOKE_EVAL`.

## Architecture

### Config: Hydra, but not a Hydra app

`train_serl.py` uses absl flags, not `@hydra.main`. Hydra is invoked programmatically inside
`workspace/base_workspace.py`, which composes `config/config.yaml` + `config/task/<exp_name>.yaml` and
returns **only the `task` subtree**. Consequences worth knowing:

- `--exp_name` selects the task YAML; `--config_override` takes repeatable Hydra overrides.
- Overrides are auto-prefixed with `task.` unless they already start with `task.`/`hydra.`
  (`_task_override`), so you write `training.batch_size=128`, not `task.training.batch_size=128`.
- `BaseWorkspace.__getattr__` forwards unknown attributes to `config.training`, which is why
  `train_serl.py` reads `workspace.batch_size`, `workspace.max_steps`, etc. with no explicit plumbing.
- Environments and agents are built by `hydra.utils.instantiate` from the `environment:` and `agent:`
  blocks — changing which env or agent a task uses is a `_target_` edit, not a code edit.

`OmegaConf` raises on missing keys even in skipped branches, so **every `wrappers.*` key must be
present in every task config**, including ones the task doesn't use. `insert_maniskill.yaml` is
deliberately standalone (no `defaults: [insert_sim]`) — inheriting would drag in a `render_spec` block
its constructor doesn't accept. The `hil` block is read with `.get()` precisely because the four
non-HIL task configs lack it.

Twelve tasks: six base (`insert_sim`, `insert_pointcloud_sim`, `insert_maniskill`, `insert_real`,
`pick_cube_sim`, `pick_place_milk`) plus six intervention variants (`insert_sim_hil`,
`insert_maniskill_hil`, `pick_place_milk_hil`, `insert_sim_human`, `pick_cube_sim_human`,
`pick_place_milk_human`) that inherit a base and add only a `hil:` block.
`pick_cube_sim.yaml` and `pick_place_milk.yaml` are standalone for the same reason as
`insert_maniskill.yaml`: inheriting another task would merge rather than replace
`environment.config`, leaving keys their constructors reject.

### Learner / actor split

Both processes run `train_serl.py:main`, which builds the same env and agent and then branches. They
communicate over [agentlace](https://github.com/youliangtan/agentlace) (`TrainerServer` on the learner,
`TrainerClient` on the actor):

- Actor → learner: transitions, pushed into two registered data stores, `actor_env` (replay) and
  `actor_env_intvn` (interventions/demos).
- Learner → actor: network weights, broadcast every `steps_per_update` steps as numpy state dicts
  (`state_dict_to_numpy` / `numpy_to_state_dict` in `algorithm/utils/train_utils.py`).
- Actor → learner: episode stats and timers, via the `send-stats` request type, logged to wandb by
  the learner only.

The learner constructs its env with `fake_env=True` — it needs the observation/action spaces to size
the buffers and networks, but never steps a simulator. This is why `infra/sim/envs/__init__.py` uses
PEP 562 lazy exports: the learner must not import mujoco or sapien.

Sampling is fixed 50/50 between the replay buffer and the demo buffer (`next_training_batch`), falling
back to replay-only when the demo buffer is empty. `cta_ratio` critic-only updates run per actor-facing
update. Checkpoints are `checkpoint_<step>.pt`; resume is automatic — `main` globs the directory and
loads the numerically highest step.

**Buffer capacities are a RAM budget, and the arithmetic is not optional.** A replay slot stores an
observation *and* a next_observation, so at two 128×128×3 uint8 cameras it costs exactly 192 KiB
(measured off a demo file, not estimated). The learner builds **two** buffers, and `_init_replay_dict`
uses `np.empty`, so pages are committed lazily — an oversized capacity does not fail at startup, it
fills RAM as the run proceeds and takes the whole desktop down with it, which reads as "training works,
then the machine freezes" rather than as an allocation error. At `actor_rate_hz: 20` that is
13.5 GiB/hour. Hence `replay_buffer_capacity: 30000` (5.5 GiB) and a separate, smaller
`demo_buffer_capacity: 15000`: the demo buffer only ever holds demonstrations plus interventions, never
the online stream. Two constraints on that second number — it must exceed the demo file's transition
count (20 human `pick_place_milk` demos are 7022 transitions, 10 `insert_sim_human` demos are 5205) or
`insert` silently wraps and overwrites the demos just loaded, and both are read with `getattr` so a task
config predating the key still works. The four base configs carrying `200000` were a leftover from when
these tasks were state-only, where the same number cost tens of MB; with images it asked for 73 GiB.

The actor's two `QueuedDataStore`s are **send queues, not buffers** — `client.update()` drains them
every `actor_update_period` steps. They were hardcoded at 50000, and agentlace's `get_latest_data`
never drops transmitted data (entries are evicted only when the deque is full), so the actor pinned up
to ~9 GiB of images alongside the learner's own buffers on the same machine. At 5000 the deque
discards oldest-first if the learner is unreachable for several minutes, which is the right failure.

The actor's `transitions` / `demo_transitions` lists exist **only** to be pickled into
`checkpoint_path/buffer` every `buffer_period` steps, and they are `deepcopy`s (which breaks the
obs/next_obs sharing, so the full 192 KiB per step). With no `--checkpoint_path` or `buffer_period: 0`
nothing ever cleared them; the actor now decides once, up front (`dump_transitions`), and says so on
stdout rather than growing silently. Those dumps are also never pruned and the learner re-reads all of
them at startup — a stale `buffer/` directory is why a run that used to survive stops surviving.

**Writing a dump is not free, and it used to land in the operator's hand.** Pickling 1000 transitions
costs a measured 244 ms against the 100 ms a 10 Hz control step allows, and `RateLimiter.sleep` does
not absorb it: an overrun takes the `else` branch and drops the backlog, so the whole duration is lost
from the episode rather than caught up. A 5.7 s freeze was observed this way (`timer/total` at step
17401 of a `pick_place_milk_human` run), with 22 steps over budget — all of them after step 13000, none
before, because this is memory-pressure-driven rather than per-step cost. `_BufferDumpWriter` moves the
pickle onto one background thread; `submit` hands the list over in 0.063 ms and rebinds. Its queue is
`maxsize=2` and **drops rather than blocks** — a resume convenience must never steal time back from a
human — and `buffer_keep_last` prunes to the newest N so the directory stops growing at ~11 GB/hour.
That write volume is the actual mechanism behind "the machine gets laggy": it flushes the page cache
hard enough to push the *desktop's* own pages into swap (`code`, `gnome-shell`, measured with swap at
2.0/2.0 GiB and `allocstall_movable` at 83766), so what stutters is the UI, not the trainer.

The two halves run at very different rates, and it matters for human-in-the-loop work: measured here,
the sim actor steps at ~97 Hz against a learner doing ~8.6 updates/s, and `steps_per_update` broadcasts
weights only every ~5.8 s — by which point the actor has moved on ~565 steps. An operator's correction
is therefore diluted about 11:1 before any gradient sees it. `--actor_rate_hz` (or `training.actor_rate_hz`,
read via `getattr` so tasks that are never hand-driven need not carry it) paces the rollout against the
wall clock; `pick_place_milk` sets 10 Hz, matching its own `control_freq` so wall-clock and simulated
time do not drift apart.

That rate belongs to the **task**, not to the teleoperation config, because three separate loops step
the same env and must agree: the training actor, `record_demo`, and `test_intervention`. All three
resolve it through `infra/utils/rate_limit.py` (`RateLimiter` + `resolve_rate_hz`). They were not
always consistent, and the failure is silent — `test_intervention` ran flat out at ~79 Hz with the HUD
on and ~127 Hz without (and varying with machine load) while the actor ran at 20, so an operator tuned
gains against an arm moving four times faster than the one they would actually train with. Worse,
`record_demo` writes files: the same hand motion sampled at two rates is two different action
distributions over identical physics, and `next_training_batch` samples 50/50 from the demo and replay
buffers. Pacing changes only *when* a step is taken — a simulator step advances a fixed slice of
simulated time either way — so it never alters the physics, only how the trajectory is sampled.
`RateLimiter.reset()` exists for the same reason the deadline does: an `env.reset` blocks far longer
than a step, and without it the limiter treats that as a backlog and sprints the start of each episode.
Evaluation is deliberately left unpaced — no human is in that loop.

`--pretrain_steps` runs `SACAgent.update_bc` over the demo buffer before online training and writes a
checkpoint that both processes then load. It exists because the startup loop waits on the **replay**
buffer, not the demo buffer, so demonstrations otherwise reach the policy only after the actor has
already spent its first ~100 steps driving a randomly initialised network. Handing the warm start over
through the network broadcast does not work — the initial publish happens after that wait, and a zmq
subscriber misses whatever was published before it connected.

`update_bc` fits **both** the mean and the std, and the reason is measured rather than assumed: fitting
the mean alone takes `mean|mode|` from 0.43 to 0.03 while leaving `scale` at its initialised 0.2–2.3, so
every sampled action still saturates tanh and the arm still thrashes — the pretraining looks like it did
nothing. Note also that the reverse fix is not available: sweeping `std_max` from 5.0 to 0.1 moves
`P(|a| > 0.9)` from 26% to 1.6% but leaves `mean|mode|` at 0.413 throughout. The loss regresses
`dist.mode()` rather than maximising a likelihood because `TanhNormal.log_prob` runs `atanh`, and the
gripper channel is a latch that demonstrations drive to exactly ±1.

**A warm start does not survive online SAC, and that is left alone deliberately.** A pretrained policy
goes from scale 0.10 / `|loc|` 0.03 to scale 0.76 / `|loc|` 0.84 within 300 updates. The cause is this
task's sparse reward — nearly every demo transition carries reward 0, so the critic learns a Q that is
almost flat in the action (−3.730 for the policy's own action, −3.731 for the demonstrated one, −3.710
for a large random one), and maximising something that flat walks the policy to the bounds. Pretraining
the critic longer does not fix it; the signal is not in the data.

A BC term on the online actor loss was tried and **removed**. The paper has no such term: its actor
loss is plain SAC with entropy regularisation (eq. 2), demonstrations only populate the demo buffer,
and it argues explicitly against regularising toward them — DAPG "performs similarly to the BC
policies" and underperforms on reactive tasks (§4.5). Staying unconstrained is how the policy is meant
to surpass the demonstrator. Early flailing is the paper's expected state, not a defect: its
intervention rate *starts* at 0.4–0.9 of timesteps and decays to 0 over 1–2.5 h (Fig. 4). `update_bc`
is kept only as a convenience for the first few minutes of teleoperation.

`--pretrain_steps` counts towards the step number, so a resumed run needs `training.max_steps` above it;
the learner raises rather than silently iterating over an empty range.

### Observation pipeline

Envs emit nested `{"state": {...}, "images": {...}}`. Wrappers are applied in
`workspace/serl_workspace.py` in a fixed order and each one is config-gated:

```
raw env
  → ExpertIntervention | SpacemouseIntervention   (innermost: sees privileged sim state, base frame)
  → TeleopHUD                                     (only when hil.hud and a display exist)
  → RelativeFrame                                 (rotates obs/action into the TCP frame at reset)
  → Quat2RotvecWrapper
  → SERLObsWrapper                                (flattens proprio_keys into a single "state" vector,
                                                   lifts images to top level)
  → ChunkingWrapper                               (obs_horizon stacking)
```

The expert is mounted *innermost* on purpose: scripted experts read privileged simulator state and both
observe and act in the base frame, before `RelativeFrame` rotates anything. `TeleopHUD` sits in the one
slot where both things it renders are still readable — `info["intervene_action"]` is still base-frame,
and `obs["images"]` is still a nested dict without a horizon axis. `SERLObsWrapper` is what makes
`training.proprio_keys` and `training.image_keys` meaningful — the workspace validates after wrapping
that every `image_keys` entry survived into the observation space.

Cartesian tasks (`insert_sim`, `insert_pointcloud_sim`, `insert_real`, `pick_cube_sim`,
`pick_place_milk`) use `RelativeFrame` + `Quat2RotvecWrapper` and a 6/7-dim action.
`insert_maniskill` is **joint-space** (9-dim `qpos` obs, 8-dim `pd_joint_delta_pos` action) and must
have those two wrappers off — they hard-require a 6-dim action and a 7-dim `tcp_pose`.

`pick_cube_sim` and `pick_place_milk` are the two tasks with a **gripper dimension** (7-dim: xyz,
rotvec, gripper). Both Cartesian wrappers only touch `action[:6]` and `tcp_pose`, so the seventh
dimension passes through them untouched — verified, not assumed. Both use the same repo-wide
convention, and it is **latched, not integrated**: `action[6] < -0.5` closes, `> 0.5` opens, and
anything between holds. A grasp has to survive a long carry during which the operator pushes only
translation and the gripper channel reads ~0; an integrating channel would drift open exactly then.
Those two thresholds live in `algorithm/utils/gripper.py`, not in the envs, because the grasp critic
below has to label recorded actions with exactly the semantics the env executed — a drift there would
mislabel training data rather than raise. That module is deliberately dependency-free: `algorithm/`
and `infra/` import nothing else from each other, which is what keeps the learner's `fake_env=True`
path from loading a simulator.

### The gripper is trained by a separate DQN, not by the Gaussian policy

`pick_place_milk` (and its `_hil` / `_human` variants) set `agent.grasp_critic: true`, which splits
the action space into the paper's two MDPs (§3.3, eq. 3): the continuous policy and critic drop to
**6 dims**, and the gripper is a discrete `{close, stay, open}` head trained by **double DQN** —
argmax from the online net, value from the target net, Polyak-averaged target. `sample_actions`
reassembles the 7-vector, so nothing outside `SACAgent` knows. The other ten tasks omit the key and
are bit-identical.

The mismatch this fixes is measurable. Inside the `±0.5` dead zone `Q` is flat in `a₆`, so there is no
gradient; evaluation's `argmax` path takes `tanh(loc)`, which almost never crosses the threshold, so
the policy simply never opens the gripper; and with sparse reward at `discount 0.97` over a 600-step
episode the grasp decision receives `0.97^600 ≈ 10⁻⁸` of the success signal. In the recorded demos
**98.56%** of transitions are `stay` (8469 of 8593, against 55 close and 69 open).

Three consequences worth knowing:

- **Checkpoints are not interchangeable** with a run that had it off — `mean_layer` is 6 wide, not 7,
  and `target_entropy` moves from -3.5 to -3.0. A shape mismatch raises even under `strict=False`.
- **Demos must carry `grasp_penalty`.** The env emits it in `info`; `record_demo` stores `infos`
  wholesale, so new recordings carry it automatically. Older files raise with the re-record command
  rather than a bare `KeyError` from inside the buffer.
- `SACAgent.load_state_dict` guards the grasp keys with `.get()`. Without that the learner would train
  a gripper policy the actor **silently discards** — no error anywhere, gripper frozen at init.

`update_bc` warm-starts the head with a class-weighted cross-entropy. Measured on a synthetic task at
the real class ratio, a 100-step budget collapsed to all-`stay` in 1 of 3 seeds unweighted versus 3 of
3 correct weighted, and by 200 steps both converged — so the weighting buys reliability at a short
`--pretrain_steps`, not correctness. The weights come from a *running* class count rather than
per-batch: at batch 64 about 63% of batches contain no `open` sample at all, so per-batch weights
alternate between ignoring a class and hitting it with a ~60× spike. Watch `bc_grasp_nonstay_frac`
and `grasp_action_nonstay_frac` — batch accuracy is near-meaningless here, since predicting `stay`
everywhere scores ~1.0 honestly.

The **grasp penalty** (`environment.config.grasp_penalty: -0.05`) enters *only* the DQN's Bellman
target, never `reward`, so the arm is not billed for the gripper's decisions. That follows upstream;
the paper's own two-MDP formulation shares one `r` and would imply folding it into both, but it never
says which and never gives a magnitude. It is charged on a **latch flip**, not on finger position —
upstream thresholds position (open > 0.9) because a real Franka gripper is effectively binary, whereas
robosuite's integrates at `speed=0.2` and sits at 0.52 right after a reset, which breaks a position
threshold in both directions at once. A clean demo costs -0.10 total against a +1.0 success.

One PyTorch trap, found by test rather than by reading: under `autocast`, a `no_grad` forward through
a module caches its downcast weights **without** grad, and a later forward through the same module
reuses that cache — so the loss comes back with `requires_grad=False` and `backward()` raises
"element 0 of tensors does not require grad". The continuous critic is immune only because its target
reads `critic_target`, a separate module; double DQN necessarily queries the online net twice. Hence
`_grasp_critic_loss_fn` runs both online forwards *first* and builds the target with `.detach()`.

Each simulator's own actuator convention differs from that, in a different way, and both are
measured rather than assumed:

- MuJoCo's `actuator8` is inverted with respect to intuition: `ctrl=0` is **closed** and `ctrl=255`
  **open**. `panda_pick_cube_gym_env.py` names the two constants for this reason.
- robosuite's `PandaGripper` is inverted *and* rate-based: it reads only `sign(action)` and
  integrates at `speed=0.2`, so `+1` closes over ~5 steps. `robosuite_pick_place_gym_env.py` holds
  the latched command asserted every step, which is exactly what drives an integrating actuator to
  its endpoint and keeps it there.

`pick_cube_sim`'s reward requires the cube to be *held* at the goal — contact with both fingers, not
just proximity — so a cube thrown through the goal sphere does not score. Grasp detection tests
contact geoms rather than finger width, because the fingers close to roughly the cube's width whether
or not anything is between them. `pick_place_milk` inverts that requirement: robosuite scores only
once the carton is in its bin quadrant **and** the gripper is more than 4.24 cm away, so a
demonstration must release and retreat.

`pick_place_milk` **pins the carton's start pose** (`object_init_pos: [0, 0]`, `object_init_rot: 0`).
Stock robosuite scatters it over the whole of bin1 — measured, 19 cm × 26 cm of travel and a full
circle of yaw — which is a wide state distribution to learn one grasp over, and is why the task was
slow to converge; with it pinned the scripted expert finishes in ~116 steps against 149–176 before.
`object_init_jitter` / `object_init_rot_jitter` are half-widths for a curriculum (0 pins exactly), and
setting the two poses to `null` restores the stock scatter. Two details that were found by running it,
not by reading:

- The pinned object gets its **own sampler**, prepended ahead of the stock one. `single_object_mode=2`
  still places the other three objects in the same bin and the sampler rejects overlapping draws, so
  pinning all four to one point makes that rejection unsatisfiable — it raises `RandomizationError`.
- That sampler sets `ensure_object_boundary_in_range=False`. The stock inset shrinks each range by the
  object's radius, which turns a zero-width range negative and makes `np.random.uniform` raise
  `high - low < 0`. Keeping the carton inside the bin is therefore the caller's job: stay within
  |x| ≤ 0.106, |y| ≤ 0.156 including jitter (bin half-extent less the carton's 0.0385 m radius).

The pose is applied after `robosuite.make` rather than passed to it, because `PickPlace` accepts no
`placement_initializer` and `_load_model` overwrites one unconditionally.

The two pick tasks differ in what they let the *policy* see, and the difference is deliberate.
`pick_cube_sim` puts `cube_to_goal` in `proprio_keys` — it is a plumbing test, and learning object
pose from pixels is not what it is validating. `pick_place_milk` does not: its `proprio_keys` are
robot proprioception only (`tcp_pose`, `tcp_vel`, `tcp_force`, `tcp_torque`, `gripper_pose`), every
one of which `infra/real/envs/franka_env.py` also publishes. Its env still computes `object_pose` and
`object_to_goal`, but only the scripted expert reads them: a policy trained on an object pose no real
setup can supply would not transfer. When adding a sim task meant to reach hardware, check new
`proprio_keys` against `franka_env`'s observation space rather than against what the simulator
happens to expose.

`ExpertIntervention` (new, config-driven) and `SpacemouseIntervention` (legacy, used by `insert_real`)
coexist in `infra/wrappers/intervention.py`. The new one is a `gym.Wrapper` rather than an
`ActionWrapper` — a scripted expert has to see observations and be told when an episode starts — and it
never appends a gripper dimension, which is what lets it serve the 8-dim joint-space task too.

### Human-in-the-loop

The entire contract is one info key: when the expert takes over, the wrapper sets
`info["intervene_action"]`, and `train_serl.py:actor` stores that action instead of the policy's *and*
inserts the transition into the demo buffer as well as the replay buffer. Given the 50/50 sampling,
corrections are weighted double with no extra code.

Experts implement the `Expert` protocol (`infra/experts/base.py`) and are registered by name in
`infra/experts/__init__.py`, imported lazily so `import infra.experts` never pulls in `pyspacemouse`
(raises with no HID device) or `mani_skill` (optional extra). Five are registered: `spacemouse`,
`keyboard`, `scripted_insert_sim`, `scripted_insert_maniskill`, `scripted_pick_place_milk`.
`hil.expert` selects one and `hil.expert_kwargs` is passed to its constructor.

`hil.trigger` decides when to take over:

| mode | intervenes when | used for |
| --- | --- | --- |
| `always` | every step | bootstrapping demos with `record_demo` |
| `manual` | `‖a_expert‖ > hil.manual_deadband` (default `1e-3`) | SpaceMouse, keyboard |
| `disagreement` | `‖a_policy − a_expert‖ > hil.disagreement_threshold` | scripted experts |

`disagreement_threshold` is **per-task and silently degenerate at both ends** — too low and this is
behavior cloning, too high and it's plain SERL. Shipped values (1.95 Cartesian, 2.32 joint-space) are
medians of the gap distribution measured against a random policy. `min_takeover_steps`,
`max_intervention_ratio`, and `intervention_decay_steps` shape `disagreement` to look human (nobody
lets go after one step; the budget decays to zero so the policy is progressively left to fail alone).
All three are **inert under `manual`** — there the human decides.

`hil.expert_frame` picks the frame the expert's action is expressed in, and the right value differs by
expert kind. `base` (the default) sends the action through untouched, which is what the scripted
experts need — they compute deltas from world-frame geometry, and rotating those would be wrong twice
over. `tcp` rotates the two leading triplets so a hand-held device feels attached to the gripper rather
than to the room. The gripper channel is never touched, and the rotation is skipped when the
observation has no 7-dim `tcp_pose`, so the joint-space task is unaffected. Note the trigger is judged
on the expert's raw output, *before* the rotation: `manual_deadband` is a property of how hard the
operator is pushing.

Two details of `tcp` are load-bearing, and both were found by measuring rather than by reasoning:

- **It is yaw-only.** Using the wrist's full orientation ties the lift axis to the tool's z, which
  points *downwards* whenever the gripper is aimed at the table — so a push upwards drives the arm
  into the bin. Only the horizontal axes follow the wrist; up stays world-up. The yaw is recovered
  from the tool x axis projected onto the horizontal plane, not from an euler decomposition, which is
  singular exactly where these tasks spend their time.
- **`hil.expert_frame_yaw` corrects the flange offset.** A gripper's body axes come from how the arm
  was assembled: the Panda's tool x runs along world **+y**, so a raw mapping swaps forward with
  sideways. A *swap* rather than a mirror is the signature of this, and `-90` squares it up on the
  Panda. Re-measure for another arm.

One trap when reading the pose back: `ExpertIntervention` must **copy** `tcp_pose` out of the
observation, not hold a reference. `RelativeFrame` rewrites that entry in place and `Quat2RotvecWrapper`
shortens it from 7 to 6, so a stored reference reads back as a 6-vector of near-zeros — which silently
disables the rotation rather than raising.

`record_demo.py` drives the env with zero actions and records whatever the expert overrides them with.
That is why the scripted flow forces `trigger=always`. A human is not pushing on every step, so use
`DEMO_TRIGGER=manual` — idle steps then record as the zero actions that genuinely executed.
`train/test_intervention.py` is the tuning counterpart: same wrapper stack, same zero policy action,
but it prints what the expert emitted next to what the environment ran and **writes nothing at all**.
Use it to check a device's feel or a deadband before committing to a recording session.

The scripted experts exist so HIL is reproducible without a SpaceMouse. The SpaceMouse path shares all
code beyond the expert itself; its device layer (a `SpaceMouse Wireless BT`, product id `0xC63A`) has
been opened and read successfully on this branch, but no *training run* has been driven by hand end to
end, so the tuning in the `_human` configs is reasoned rather than measured.

### Teleoperation (keyboard / HUD)

`insert_sim_human` is the hand-driving config: `trigger: manual`, keyboard expert, and a raised episode
budget (`max_episode_length: 1000`, `time_limit: 60.0` — the stock 100 steps at 50 Hz is two seconds of
simulated time, far too short for a human). `EXPERT=spacemouse` switches devices.

`pick_cube_sim_human` is the SpaceMouse config, and the second human-drivable task: `trigger: manual`,
`expert: spacemouse`, with a higher `pos_scale` (0.4) and lower `expo` (1.5) than the insertion configs
because the work is travel rather than sub-millimetre alignment. Both puck buttons are **edge-triggered
toggles** (see below), at `gripper_scale: 1.0` so a click clears the ±0.5 latch
thresholds. `pick_place_milk_human` is the third, on the same puck bindings at `pos_scale` 0.96; its
operator has to remember to release **and pull back** 4.24 cm, or the attempt is discarded as a failure
however good the placement looked. That gain is sized against the **wall clock**, not against the
recordings, and the two are not the same thing once the loop is paced: speed is per-step displacement
times rate, so capping the rate at 20 Hz cut the arm to a quarter of the metres per second it used to
cover and teleoperation felt heavy. Measured, the unthrottled ~79 Hz loop delivered 0.250 m/s, and at
20 Hz `pos_scale` 0.25 is 0.063 m/s, 0.5 is 0.130, and 0.96 restores the original feel at 0.249. Those
figures were taken at 20 Hz; the task has since dropped to `actor_rate_hz: 10` alongside
`control_freq: 10`, so a step covers twice the distance and the arm moves about twice as fast as they
say — re-time before trusting one.
Translation takes this without complaint (the OSC's achieved-to-commanded ratio is a flat 0.26 from
0.25 to full deflection, std/mean 0.007), but **rotation saturates** — 0.23 at 0.15, 0.19 at 0.6, 0.14
at full — so matching the old 1.382 rad/s needs `rot_scale` 0.85 rather than a proportional scaling,
and 1.0 would buy 4% more rotation for a visibly less linear response.

The consequence worth knowing: recovering the unpaced feel spends nearly the whole action space, so
`environment.config.action_limit` is now `null`. It clips the human's action as well as the policy's,
so it can never sit below `pos_scale`/`rot_scale` — a cap under the gains silently limits the operator
and pushing the puck harder does nothing — and with the gains at 0.96/0.85 against a ±1 action space,
every admissible cap is within a few percent of no cap at all. Left as a number it would read as if the
policy were still being held back. **The remaining lever for speed is `control_freq`** (with
`training.actor_rate_hz` moved to match), which buys it through rate rather than per-step displacement.
It also sets `invert_xy: true`, which both human experts accept: robosuite's
stock scene camera faces the robot and mirrors the horizontal plane, so without it a push to the right
sends the arm left in the picture the operator is watching. Correcting it in the expert rather than by
moving the camera keeps that clear view of both bins, and touches only the human's own actions —
scripted demonstrations and the policy's action space are unchanged. `insert_maniskill` remains
undrivable by hand — 8-dim joint-space, would need an IK layer.

### The SpaceMouse's two buttons are edge-triggered toggles

Left toggles the gripper (click to close, click again to open); right toggles the trigger in force
between `manual` and `always`, i.e. between correcting the policy and driving the whole episode. Both
are counted as **rising edges in the free-running reader process**
(`SpaceMouseExpert._read_spacemouse`), not sampled as levels in the control step: a control step is
100 ms at 10 Hz and a crisp click fits between two samples. Under the old hold-to-act binding a missed
sample only meant "hold it a moment longer"; under a toggle it leaves the operator's next click
inverted, so the edge cannot be allowed to slip. The reader also zero-fills `buttons` on any read
exception, which would forge a release/press pair — a counter cannot.

Three consequences that are load-bearing:

- **The gripper command is held asserted every step, but only the flip is recorded.** Both envs latch,
  so a pulse would be enough on its own — but on the steps the policy is steering it drives that
  channel too, and with `grasp_critic` its greedy argmax is exactly −1/0/+1, which clears the same ±0.5
  threshold. A pulsed grasp is released by the policy one step later. `ExpertIntervention._hold_gripper`
  therefore overrides `action[-1]` even on non-intervention steps.

  What goes to the *environment* and what goes into `info["intervene_action"]` / `info["total_action"]`
  are deliberately not the same vector, and conflating them silently destroyed a training run. A
  SpaceMouse holds its latch at ±1 on **every** step (`gripper_scale: 1.0`), so re-asserting that into
  the recording erases `stay` from the data entirely: a 20-demo `pick_place_milk_human` file came out
  41.68% close / 58.32% open / **0.00% stay**, against the ~98.6% stay this task actually has, with
  `action[6]` taking only two distinct values. `_grasp_class_weights` then reads a zero count for stay
  and weights it 1303×, teaching the head that holding never happens — the measured symptom is
  `grasp_action_nonstay_frac ≡ 1.000`, a gripper that flips open/closed every step and carries nothing
  (1 successful grasp in 91 episodes; 6 in 240 across four runs). The wrapper now records `stay` unless
  the command actually flips the latch it is tracking, while still asserting ±1 to the integrating
  actuator. `record_demo` prints the class histogram on save and warns below 80% stay, because this
  failure is invisible until you read it off wandb hours later.
- **`manual`'s deadband measures `expert_action[:6]` only.** A held gripper latch would otherwise pin
  the norm above any deadband forever and the human could never hand motion control back. (The
  keyboard expert still has this bug: `keyboard.py` latches `_gripper` to ±1 permanently, so on a
  7-dim task one press of space holds control for the rest of the run.)
- **The mode survives `env.reset`, the gripper belief does not.** The mode is the operator's standing
  choice and silently reclaiming control at an episode boundary is exactly what they cannot see
  coming; the latch, by contrast, is reset to open by both envs, so the wrapper resyncs the expert
  through `sync_gripper` or the first click of the new episode is swallowed correcting a stale belief.
  A `disagreement` config refuses the switch outright and says so — its budget keeps accruing while
  another mode is engaged, so a round trip would return to a mode whose budget is already spent.

The keyboard expert (`infra/experts/keyboard.py`) picks one of two backends at construction. With
`pynput` installed and an X server reachable, an action lasts exactly as long as the key is held.
Otherwise it reads the teleop window's `waitKey`, which reports key *repeats* rather than holds, so a
press is latched for `expert_kwargs.sticky_steps` control steps and fades out — jogging rather than
holding. With neither backend reachable it returns zeros forever and never intervenes.

`hil.hud: true` opens one `TeleopDisplay` (`infra/utils/teleop_display.py`) shared by the HUD wrapper
and the cv2 keyboard backend — a cv2 window only reports keys to the thread that owns it, so a second
window would starve one of the two. It is skipped when `DISPLAY`/`WAYLAND_DISPLAY` is unset, because
Qt aborts the whole process inside `cv2.imshow` before any catchable Python exception exists.

`hil.hud_render_size` (640 in both SpaceMouse configs) makes `TeleopHUD` **re-render** the operator's
view at that resolution through the env's own `render_camera`, instead of upscaling the policy's
128×128 observation. The distinction matters because the on-screen blur is a *resolution* limit, not
an interpolation artefact: at 128 the wrist camera's own gripper fingers are a few pixels across, and
no filter recovers what was never rendered — swapping `INTER_NEAREST` for a smooth kernel removes the
jaggies and adds no detail. It doubles as the window size, since tiles sit side by side at their own
resolution: 640 gives a 1280×766 window. Measured on `pick_place_milk` against the 100 ms a 10 Hz loop
allows, the two extra renders cost +5.2 ms/step at 384, +6.6 at 640, +10.3 at 720 and +16.1 at 768 —
640 is where that curve turns.

Three things this deliberately does not change: the policy still observes 128×128 (`image_size` is
rejected at any other value — the SAC pixel encoder is built for it), the replay buffers still cost
192 KiB per slot, and recorded demos are unaffected. Set it to `0` to fall back to upscaling, which is
also what a task whose env has no `render_camera` gets. `TeleopDisplay` scales a tile only up to
`128 * hud_scale`, so an already-large frame is passed through rather than blown past the screen edge.

Two companion keys keep the two audiences apart. `hil.hud_cameras` **pins** the operator's views by
name instead of following the observation, so adding a camera to `training.image_keys` changes what
the network sees without rearranging the window an operator has learned to fly by — the top row stays
the same two feeds at the same size. `hil.hud_show_policy_view: true` then draws the policy's own
frames beneath them at their true 128×128, captioned `POLICY INPUT`, and *that* row does follow
`image_keys`. Deliberately not upscaled: the row exists to show the gap between what the operator is
steering by and what the network can possibly learn from, and enlarging it would hide exactly that.
Both default off, so a task that sets neither behaves as before.

### Real robot

`SERL actor → Franka HTTP bridge (infra/hardware/robot/franka_server.py, Flask) → ROS 2 Cartesian
impedance controller → FR3`. The bridge owns the controller lifecycle — it launches
`impedance.launch.py`, switches to `joint.launch.py` for joint resets, and stops them on shutdown. Do
not launch the impedance controller separately alongside it. The ROS 2 side lives in a separate repo,
[serl_controller_ros2](https://github.com/liusong-0086/serl_controller_ros2).

All poses in this project are `[x, y, z, qx, qy, qz, qw]` in the Franka `base` frame.
`abs_xyz_limit_low/high` in `insert_real.yaml` are *positive distances relative to `target_pose`*, not
absolute base-frame coordinates. See the README's field table for the rest of `insert_real.yaml`.

## Conventions that bite

- Headless rendering is auto-configured in `infra/sim/envs/__init__.py`: if there's no `DISPLAY` and no
  `MUJOCO_GL`, it probes for EGL then OSMesa. Don't hardcode a backend — a missing library fails at
  `import mujoco` with an opaque OpenGL error. `mani_skill` imports mujoco transitively, so this
  governs the ManiSkill tasks too.
- Adding a new task = a new `config/task/*.yaml` (discovered by glob via `available_tasks()`) plus, if
  needed, a new `_target_` env class. No registration code.
- The ManiSkill demo dataset drops each episode's final frame and did not record seeds, so
  demonstrations there **cannot be replayed** in the environment. They are off-policy data only.
- In an `insert_sim` demo file roughly 40% of recorded actions are exactly zero, in contiguous runs.
  That is correct: the expert closes the loop on the MuJoCo mocap setpoint and the delta is zero while
  the impedance servo catches up.
- robosuite renders images **upside down** (`macros.IMAGE_CONVENTION` defaults to `"opengl"`), raises
  `done` from the horizon alone so success never terminates an episode, and returns the old-gym
  `reset()` obs / 4-tuple `step()`. All three fail silently rather than crashing —
  `robosuite_pick_place_gym_env.py` flips the frames, polls `_check_success()`, and adapts the API.
  Its scene also has no `front` camera, so evaluation needs `--eval_video_main_camera=agentview`.
- `_zero_observation` in the third-party env wrappers must build its shapes from a module constant,
  not from `self.observation_space`: `Quat2RotvecWrapper` rewrites that space's `tcp_pose` entry
  **in place**, and an all-zero quaternion is not a rotation — both only ever break the learner,
  which is the one process that takes the `fake_env` path.
- Comments in this codebase mostly explain *why a non-obvious choice was made* (why a config is
  standalone, why an import is lazy, why a wrapper sits where it does). Match that register rather than
  narrating what the code does.
