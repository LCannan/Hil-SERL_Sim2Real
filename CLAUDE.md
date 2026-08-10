# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PyTorch reimplementation/fork of [HIL-SERL](https://github.com/rail-berkeley/hil-serl): distributed
SAC for robot manipulation, with human-in-the-loop interventions. Runs on MuJoCo sim, ManiSkill/SAPIEN
sim, and a real Franka FR3. The upstream project is JAX; this one is not — there is no Flax/Optax code.

Compared to upstream HIL-SERL: no reward classifier, no BC or gradient-penalty regularization, no
pre-training phase.

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
the ManiSkill scene** — pass `render_camera` there or evaluation crashes.

Demo collection: `uv run python -m train.record_demo --exp_name=<task> --successes_needed=N`.

Two wrapper scripts cover the full flow and are the fastest way to smoke-test a change:

```bash
./scripts/run_hil_serl.sh insert_sim all        # demos -> short learner+actor -> eval
./scripts/run_insert_maniskill.sh prepare       # download HF dataset + convert demos
```

Both forward extra args to `train_serl`, so
`./scripts/run_hil_serl.sh insert_sim learner --config_override=training.batch_size=128` works.

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
pre-existing task configs lack it.

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

### Observation pipeline

Envs emit nested `{"state": {...}, "images": {...}}`. Wrappers are applied in
`workspace/serl_workspace.py` in a fixed order and each one is config-gated:

```
raw env
  → ExpertIntervention | SpacemouseIntervention   (innermost: sees privileged sim state, base frame)
  → RelativeFrame                                 (rotates obs/action into the TCP frame at reset)
  → Quat2RotvecWrapper
  → SERLObsWrapper                                (flattens proprio_keys into a single "state" vector,
                                                   lifts images to top level)
  → ChunkingWrapper                               (obs_horizon stacking)
```

The expert is mounted *innermost* on purpose: scripted experts read privileged simulator state and both
observe and act in the base frame, before `RelativeFrame` rotates anything. `SERLObsWrapper` is what
makes `training.proprio_keys` and `training.image_keys` meaningful — the workspace validates after
wrapping that every `image_keys` entry survived into the observation space.

Cartesian tasks (`insert_sim`, `insert_pointcloud_sim`, `insert_real`) use `RelativeFrame` +
`Quat2RotvecWrapper` and a 6/7-dim action. `insert_maniskill` is **joint-space** (9-dim `qpos` obs,
8-dim `pd_joint_delta_pos` action) and must have those two wrappers off — they hard-require a 6-dim
action and a 7-dim `tcp_pose`.

### Human-in-the-loop

The entire contract is one info key: when the expert takes over, the wrapper sets
`info["intervene_action"]`, and `train_serl.py:actor` stores that action instead of the policy's *and*
inserts the transition into the demo buffer as well as the replay buffer. Given the 50/50 sampling,
corrections are weighted double with no extra code.

Experts implement the `Expert` protocol (`infra/experts/base.py`) and are registered by name in
`infra/experts/__init__.py`, imported lazily so `import infra.experts` never pulls in `pyspacemouse`
(raises with no HID device) or `mani_skill` (optional extra). `hil.expert` in config selects one.

`hil.trigger` decides when to take over: `always` (demo bootstrapping), `manual` (`‖a_expert‖ > 1e-3`,
i.e. a human pushing a SpaceMouse), `disagreement` (`‖a_policy − a_expert‖ > threshold`, for scripted
experts). `disagreement_threshold` is **per-task and silently degenerate at both ends** — too low and
this is behavior cloning, too high and it's plain SERL. Shipped values (1.95 Cartesian, 2.32
joint-space) are medians of the gap distribution measured against a random policy.

The scripted experts exist so HIL is reproducible without a SpaceMouse; the SpaceMouse path shares all
code beyond the expert itself but **has not been tested against real hardware on this branch**.

### Real robot

`SERL actor → Franka HTTP bridge (infra/hardware/robot/franka_server.py, Flask) → ROS 2 Cartesian
impedance controller → FR3`. The bridge owns the controller lifecycle — it launches
`impedance.launch.py`, switches to `joint.launch.py` for joint resets, and stops them on shutdown. Do
not launch the impedance controller separately alongside it. The ROS 2 side lives in a separate repo,
[serl_controller_ros2](https://github.com/liusong-0086/serl_controller_ros2).

All poses in this project are `[x, y, z, qx, qy, qz, qw]` in the Franka `base` frame.
`abs_xyz_limit_low/high` in `insert_real.yaml` are *positive distances relative to `target_pose`*, not
absolute base-frame coordinates.

## Conventions that bite

- Headless rendering is auto-configured in `infra/sim/envs/__init__.py`: if there's no `DISPLAY` and no
  `MUJOCO_GL`, it probes for EGL then OSMesa. Don't hardcode a backend — a missing library fails at
  `import mujoco` with an opaque OpenGL error. `mani_skill` imports mujoco transitively, so this
  governs the ManiSkill tasks too.
- Adding a new task = a new `config/task/*.yaml` (discovered by glob via `available_tasks()`) plus, if
  needed, a new `_target_` env class. No registration code.
- The ManiSkill demo dataset drops each episode's final frame and did not record seeds, so
  demonstrations there **cannot be replayed** in the environment. They are off-policy data only.
- Comments in this codebase mostly explain *why a non-obvious choice was made* (why a config is
  standalone, why an import is lazy, why a wrapper sits where it does). Match that register rather than
  narrating what the code does.
