# SERL-Plus-Plus

![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

> This repository is built upon a fork of [HIL-SERL](https://github.com/rail-berkeley/hil-serl).

## Requirements

- Python 3.10.19
- CUDA 12.4+ recommended for GPU acceleration
- PyTorch 2.4.1+
- MuJoCo 3.4.0
- ManiSkill 3.0.1+ (optional, only for the ManiSkill task: `uv sync --extra maniskill`)
- `uv` for dependency and virtual environment management
- See `pyproject.toml` for the full dependency list

## Installation

```bash
# Clone repo
git clone <repository-url>

# Enter repo
cd serl-plus-plus

# Create and sync the virtual environment
uv sync

# Activate the environment
source .venv/bin/activate
```

## Quick Start

Training uses two processes:

- The learner owns optimization, checkpointing, logging, and the trainer server.
- The actor interacts with the environment and sends transitions to the learner.

Start the learner first, then start the actor in another terminal. If the actor
runs on another machine, pass the learner address with `--ip=<learner-ip>`.
Task definitions live in `config/task` and can be changed at launch with repeatable
`--config_override` flags.

### 1. Peg Insert Sim with RGB

![peg_insert_sim](./doc/peg_insert_sim.gif)

```bash
# Download demo data
mkdir -p demo_data
wget -P demo_data https://github.com/liusong-0086/serl-plus-plus/releases/download/demo_data/peg_insert_sim_20_demos.pkl

# Terminal 1: start learner node
uv run python -m train.train_serl \
  --exp_name=insert_sim \
  --demo_path=demo_data/peg_insert_sim_20_demos.pkl \
  --checkpoint_path=checkpoints/insert_sim \
  --learner

# Terminal 2: start actor node
uv run python -m train.train_serl \
  --exp_name=insert_sim \
  --checkpoint_path=checkpoints/insert_sim \
  --actor
```

Simulation uses off-screen rendering when no display is available. To evaluate
a saved policy without starting a learner and save one MP4 video per trajectory:

```bash
uv run python -m train.train_serl \
  --exp_name=insert_sim \
  --checkpoint_path=checkpoints/insert_sim \
  --actor \
  --eval_n_trajs=10
```

By default this evaluates the numerically latest checkpoint and writes videos
to `checkpoints/insert_sim/videos`. Use `--eval_checkpoint_step=50000` to select
an exact checkpoint or `--eval_video_dir=<directory>` to change the output.
Videos preserve every control-step frame, play at 20 FPS, and hold briefly on
the initial and terminal frames. These can be adjusted with
`--eval_video_fps`, `--eval_video_start_hold_seconds`, and
`--eval_video_end_hold_seconds`. Simulation videos place the `front` camera
above the side-by-side wrist views; use `--eval_video_main_camera=` to disable
the main view.

### 2. Peg Insert Sim with Point Cloud

![peg_insert_pointcloud_sim](./doc/peg_insert_pointcloud_sim.gif)

```bash
# Download demo data
mkdir -p demo_data
wget -P demo_data https://github.com/liusong-0086/serl-plus-plus/releases/download/demo_data/peg_insert_pointcloud_sim_20_demos.pkl

# Terminal 1: start learner node
uv run python -m train.train_serl \
  --exp_name=insert_pointcloud_sim \
  --demo_path=demo_data/peg_insert_pointcloud_sim_20_demos.pkl \
  --checkpoint_path=checkpoints/insert_pointcloud_sim \
  --learner

# Terminal 2: start actor node
uv run python -m train.train_serl \
  --exp_name=insert_pointcloud_sim \
  --checkpoint_path=checkpoints/insert_pointcloud_sim \
  --actor
```

### 3. Peg Insert ManiSkill

A SAPIEN/ManiSkill port of `PegInsertionSide-v1` with a widened hole clearance.
Unlike the MuJoCo tasks above it is **joint-space**: a 9-dim `qpos` observation
and an 8-dim `pd_joint_delta_pos` action. This matches the
[RLinf ManiSkill peg-insertion dataset](https://huggingface.co/datasets/RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ),
whose episodes can be converted into SERL demonstrations.

ManiSkill is an optional dependency. Install it with the `maniskill` extra —
note that a later bare `uv sync` **removes** extra-only packages, so keep the
flag:

```bash
uv sync --extra maniskill
```

A helper script wraps the whole flow:

```bash
./scripts/run_insert_maniskill.sh prepare   # download dataset + build demos
./scripts/run_insert_maniskill.sh learner   # terminal 1
./scripts/run_insert_maniskill.sh actor     # terminal 2
./scripts/run_insert_maniskill.sh eval 10   # evaluate, writes 10 MP4s
```

`prepare` skips the download if the dataset is already on disk and re-validates
existing demos instead of rebuilding them. Override `EXP_NAME`,
`CHECKPOINT_PATH`, `DEMO_PATH`, `NUM_DEMOS`, `SEED`, or `EVAL_MAIN_CAMERA` via
the environment; any extra arguments are forwarded to `train.train_serl`, so
`./scripts/run_insert_maniskill.sh learner --config_override=training.batch_size=128`
works. The equivalent commands by hand:

```bash
# Download the dataset (~9.2 GB) and convert 30 random episodes into demos
uv run hf download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
  --repo-type dataset \
  --local-dir demo_data/rlt-maniskill-PegInsertionSide-v1-400-succ

uv run python train/convert_lerobot_demo.py \
  --dataset_path=demo_data/rlt-maniskill-PegInsertionSide-v1-400-succ \
  --output_path=demo_data/insert_maniskill_30_demos.pkl \
  --num_episodes=30

# Terminal 1: start learner node
uv run python -m train.train_serl \
  --exp_name=insert_maniskill \
  --demo_path=demo_data/insert_maniskill_30_demos.pkl \
  --checkpoint_path=checkpoints/insert_maniskill \
  --learner

# Terminal 2: start actor node
uv run python -m train.train_serl \
  --exp_name=insert_maniskill \
  --checkpoint_path=checkpoints/insert_maniskill \
  --eval_video_main_camera=render_camera \
  --actor
```

As with `insert_sim`, evaluation renders off-screen and writes one MP4 per
trajectory to `checkpoints/insert_maniskill/videos` — no window is opened, so it
works over SSH:

```bash
uv run python -m train.train_serl \
  --exp_name=insert_maniskill \
  --checkpoint_path=checkpoints/insert_maniskill \
  --eval_video_main_camera=render_camera \
  --actor \
  --eval_n_trajs=10
```

Each frame places the `render_camera` overview above the side-by-side
`3rd_view_camera` and `wide_hand_camera` views. The same
`--eval_checkpoint_step`, `--eval_video_dir`, `--eval_video_fps`, and hold-time
flags described in section 1 apply.

The converter downsamples the dataset's 384x384 frames to 128x128 with the same
`resize_rgb` helper the environment uses, so demonstration and online
observations are processed identically.

Two things worth knowing about the dataset:

- The collector drops each episode's final frame, so an N-frame episode yields
  N-1 transitions and the success reward lands on the last transition available
  rather than on a true post-insertion state.
- Episode seeds were not recorded, and peg geometry is resampled on every reset,
  so the demonstrations cannot be replayed in the environment. HIL-SERL treats
  them as off-policy data, so this is fine — but do not expect replay to verify
  them.

`--eval_video_main_camera` defaults to `front`, which does not exist in this
scene; pass `render_camera` as shown above.

### 4. Pick Cube Sim

Grasp a 4 cm cube and carry it to a randomized 3D goal, in MuJoCo. Both the cube's
xy and the goal's xyz are resampled every reset; a translucent green sphere shows
where the goal is.

This is the only task in the repo with a **gripper**, so its action is 7-dim —
`[dx, dy, dz, drx, dry, drz, gripper]` — where `gripper < -0.5` closes,
`> 0.5` opens, and anything between holds the current state. The reward is sparse
and requires the cube to be **held at the goal**: both fingers in contact *and*
the cube within `goal_threshold` (4.5 cm) of the marker. Throwing the cube through
the sphere does not count.

```bash
# Terminal 1: learner (needs demonstrations; see section 5 to record them by hand)
uv run python -m train.train_serl \
  --exp_name=pick_cube_sim \
  --demo_path=demo_data/pick_cube_sim_human_demos.pkl \
  --checkpoint_path=checkpoints/pick_cube_sim \
  --learner

# Terminal 2: actor
uv run python -m train.train_serl \
  --exp_name=pick_cube_sim \
  --checkpoint_path=checkpoints/pick_cube_sim \
  --actor
```

`pick_cube_sim_human` is the teleoperation variant, driven by SpaceMouse — see
[Picking a cube with a SpaceMouse](#picking-a-cube-with-a-spacemouse).

A few numbers, measured rather than guessed, in case you retune it: the TCP sits
~9.9 cm above the cube's centre once the grasp closes, the grasp height is ~11.8 cm
at `hand_site`, and a competent open-loop controller completes an episode in ~63
steps against this task's 300-step budget. `goal_threshold` is 4.5 cm because a
controller that tracks the goal well still settles 3.1–3.5 cm away — the impedance
servo lags its setpoint — so a tighter threshold makes success knife-edge.

That 300-step budget is sized for a policy, not a person; `pick_cube_sim_human`
raises it to 2000 steps / 120 s for hand-driving.

Unlike `insert_sim`, this task's `proprio_keys` include `gripper_pose` and
`cube_to_goal`: without the first the policy cannot distinguish an open hand from
a closed one, and without the second it cannot know where the goal moved to.

### 5. Pick Place Milk (robosuite)

Pick a milk carton out of the source bin and place it in its quadrant of the
target bin, in [robosuite](https://robosuite.ai)'s `PickPlaceMilk`. robosuite is
a hard dependency and needs no extra install.

Like `pick_cube_sim` this task has a **gripper**, so its action is 7-dim —
`[dx, dy, dz, drx, dry, drz, gripper]` — and follows this repo's convention:
`gripper < -0.5` closes, `> 0.5` opens, anything between holds. robosuite's own
convention is the opposite sign *and* rate-based rather than latched; the
environment translates, holding the latched command asserted every step.

The reward is sparse and comes from robosuite's `_check_success()` unchanged.
**Success requires releasing the carton and pulling back:** the criterion is the
carton inside its bin quadrant *and* the gripper more than **4.24 cm** away
(`r_reach = 1 - tanh(10*d) < 0.6`). A run that places the carton perfectly and
hovers over it scores exactly zero — worth knowing before recording demos by
hand.

```bash
# Record 20 demonstrations with the scripted expert (no hardware needed)
./scripts/run_hil_serl.sh pick_place_milk demos 20

# Terminal 1: learner
./scripts/run_hil_serl.sh pick_place_milk learner
# Terminal 2: actor, with the expert intervening on disagreement
./scripts/run_hil_serl.sh pick_place_milk actor

# Evaluate the policy alone, writing MP4s
./scripts/run_hil_serl.sh pick_place_milk eval 10
```

`pick_place_milk_human` is the teleoperation variant, driven by SpaceMouse — see
[Picking a cube with a SpaceMouse](#picking-a-cube-with-a-spacemouse) for the
device setup, which is identical.

A few numbers, measured rather than guessed, in case you retune it. The scripted
expert succeeds **13/15** at 149–260 control steps, against this task's 400-step
budget — measured through the full wrapper stack with `hil.trigger=always`, which
is the only meaningful way to measure it: `RelativeFrame` rotates actions into the
TCP frame, so an expert benchmarked against the bare environment can score
perfectly there and still fly the arm out of the workspace once wrapped. Its
`disagreement_threshold` of 2.05 is the median of ‖a_policy − a_expert‖ over 1230
steps of a random policy (p25 1.67, p75 2.38).

The milk carton is **not** robosuite's stock one. That is 4 cm across and 15.8 cm
tall, which is a fiddly target for a hand on a SpaceMouse, so
`environment.config.object_xml` points at
[milk_wide.xml](infra/sim/envs/assets/robosuite_objects/milk_wide.xml) — 6.4 cm
across and 10.5 cm tall. Set `object_xml` to `null` for the stock carton.

Wider is not simply better, and the trade-off is steep: the gripper spans 8 cm,
so a wider carton leaves less clearance and demands a more precisely centred
approach. Measured, re-tuning the expert's `grasp_height` for each width, 5.6 cm
scores 10/10, 6.4 cm scores 7/8, 6.8 cm drops to 4/6 and 7.9 cm fails outright at
1/6. `grasp_height` is measured from the object's body origin and is specific to
the geometry, so re-tune and re-measure it if you change the carton.

Two robosuite conventions differ from the rest of this repo and both fail
silently rather than crashing: images come back **vertically flipped**
(`macros.IMAGE_CONVENTION` is `"opengl"`), and `done` is raised by the horizon
alone, so success never terminates an episode on its own. The environment flips
the frames and polls `_check_success()` itself.

Both cameras are left at their robosuite defaults, which means the scene camera
faces the robot and so **mirrors the horizontal plane on screen**: push a teleop
device right and the arm moves left in the picture. Moving the camera behind the
robot fixes the axes but costs the view — every placement back there either puts
the arm across the frame or pushes the bins too far away to read at 128×128. The
mirroring is corrected in the teleop experts instead, via
`hil.expert_kwargs.invert_xy` in the `_human` config, which negates the operator's
two horizontal translation axes. That affects only the human's own actions, so
the scripted expert's demonstrations and the policy's action space are untouched.
The env still exposes `agentview_pos` / `agentview_lookat` / `wrist_camera_roll`
if you want to move the cameras anyway.

The policy's `proprio_keys` are **robot proprioception only** — `tcp_pose`,
`tcp_vel`, `tcp_force`, `tcp_torque`, `gripper_pose` — every one of which
`infra/real/envs/franka_env.py` also publishes, so the observation transfers to
hardware unchanged. The environment additionally computes `object_pose` and
`object_to_goal` and leaves them out of `proprio_keys` on purpose: they are
privileged simulator state for the scripted expert to read, and a policy trained
on them would depend on an object pose no real setup can supply. The carton has
to be located from the two camera feeds.

`--eval_video_main_camera` must be `agentview` here — the bins scene has no
`front` camera, so the flag's default would crash evaluation.
`run_hil_serl.sh` passes it automatically.

### 5. FR3 Real Robot

The real-robot stack uses
[serl_controller_ros2](https://github.com/liusong-0086/serl_controller_ros2)
as the ROS 2 low-level controller. The data path is:

```text
SERL actor -> Franka HTTP bridge -> ROS 2 Cartesian impedance controller -> FR3
           <- pose / velocity / wrench / joints / Jacobian state          <-
```

Keep the emergency stop within reach, clear the workspace, and test with low
action scales and conservative stiffness values before collecting data. All
poses in this project use `[x, y, z, qx, qy, qz, qw]` in the Franka `base`
frame.

#### Install the ROS 2 controller

The controller package targets ROS 2 Humble and must be built in the same
workspace as `franka_ros2` so that `franka_bringup`, `franka_msgs`, and the
Franka hardware interfaces are available.

```bash
source /opt/ros/humble/setup.bash

mkdir -p ~/franka_ros2_ws/src
cd ~/franka_ros2_ws/src
git clone https://github.com/liusong-0086/serl_controller_ros2.git

cd ~/franka_ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build \
  --packages-select serl_franka_controllers_ros2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Before continuing, make sure the control computer can reach the robot, the FR3
is unlocked with FCI enabled, and `ping 172.16.0.2` succeeds. Replace
`172.16.0.2` below if the robot uses another address.

#### Configure the task
Edit `config/task/insert_real.yaml` for the physical setup. The most important
fields are:

| Field | Meaning |
| --- | --- |
| `server_url` | Franka HTTP bridge URL. Keep `http://127.0.0.1:5000/` when the actor and bridge run on the same computer. |
| `realsense_cameras` | Camera names and serial numbers. Each name must also appear in `training.image_keys`. |
| `image_crop` | Per-camera crop `[top, bottom, left, right]`, applied before resizing to `128 x 128`. |
| `target_pose` | Task success pose in the Franka `base` frame. |
| `reset_pose` | Cartesian pose used at the beginning of an episode. Use a normalized `xyzw` quaternion. |
| `reward_threshold` | Absolute success tolerances `[x, y, z, rx, ry, rz]` in meters and radians. |
| `action_scale` | The first two entries scale normalized translation and rotation actions in meters and radians. Keep all three entries required by the current config schema. |
| `abs_xyz_limit_low/high` | Positive distances below/above `target_pose`; together they define the Cartesian safety box. They are not absolute base-frame coordinates. |
| `random_r*_range(_neg)` | Rotation limits around `reset_pose`, also used by pose clipping when `random_reset` is disabled. The `_neg` values are lower bounds and should be negative or zero. |
| `compliance_param` | Impedance values used during interaction and training. |
| `precision_param` | Stiffer impedance values used only while returning to `reset_pose`. |
| `joint_reset_period` | Run a joint-space reset every N episodes; `0` disables periodic joint resets. |

#### Start and verify the Franka bridge

The bridge owns the controller lifecycle: it launches `impedance.launch.py` at
startup, enables the Jacobian stream, temporarily switches to
`joint.launch.py` for joint resets, and stops its ROS 2 launch processes on
shutdown. Do not separately launch the impedance controller while the bridge
is running.

```bash
cd ~/serl-plus-plus
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
source .venv/bin/activate

python -m infra.hardware.robot.franka_server \
  --robot_ip=172.16.0.2 \
  --robot_type=fr3 
```

#### Collect demonstrations

Keep the bridge running. From the SERL-Plus-Plus environment, collect successful
SpaceMouse trajectories:

```bash
cd ~/serl-plus-plus
uv run python -m train.record_demo \
  --exp_name=insert_real \
  --successes_needed=20
```

Only successful episodes are saved. The resulting file is written under
`demo_data/`; use that exact path for learner startup.

#### Train on the robot

Start the learner first:

```bash
uv run python -m train.train_serl \
  --exp_name=insert_real \
  --demo_path=demo_data/<insert_real_demo_file>.pkl \
  --checkpoint_path=checkpoints/insert_real \
  --learner
```

Then start the actor on the computer connected to the robot, cameras, and
SpaceMouse:

```bash
uv run python -m train.train_serl \
  --exp_name=insert_real \
  --checkpoint_path=checkpoints/insert_real \
  --actor \
  --ip=<learner-ip>
```

### 5. Human-in-the-Loop SERL

The tasks above are pure SERL: the policy drives every step. HIL-SERL adds an
expert who takes over when the policy is about to fail. Intervened steps are
written to the replay buffer *and* to the demo buffer, so the learner's fixed
50/50 sampling weights them double — corrections become new demonstrations with
no extra code.

The intervention contract is a single key. Whenever the expert takes over, the
wrapper sets `info["intervene_action"]` to the action it actually executed, and
`train_serl.py` stores that instead of the policy's action.

Running this normally needs a SpaceMouse. To make it reproducible without
hardware, the expert is an interface with several implementations — a
SpaceMouse, a keyboard, and a scripted controller — selected by config:

```bash
./scripts/run_hil_serl.sh insert_sim demos 20   # scripted expert records 20 demos
./scripts/run_hil_serl.sh insert_sim learner    # terminal 1
./scripts/run_hil_serl.sh insert_sim actor      # terminal 2, expert intervenes
./scripts/run_hil_serl.sh insert_sim eval 10    # policy alone, writes 10 MP4s
./scripts/run_hil_serl.sh insert_sim all        # smoke test: all of the above
```

Substitute `insert_maniskill` for the ManiSkill task; the script picks the
`_hil` config variant and the right render camera automatically. Nothing is
downloaded — the demonstrations in step one are generated by the scripted
expert.

In an `insert_sim` demo file roughly 40% of the recorded actions are exactly
zero, in contiguous runs. That is correct: the expert closes the loop on the
MuJoCo mocap setpoint, and once the setpoint is on target the delta is zero
while the impedance servo spends many steps catching up. The ManiSkill expert
commands joint deltas and has no such plateaus.

The scripted expert reads privileged simulator state (the MuJoCo mocap setpoint,
the ManiSkill peg and goal poses). It stands in for a human, who likewise knows
things the policy's cameras do not; it never enters the agent's observation.
Measured success rate driving from reset: 8/10 on `insert_sim`, 7/8 on
`insert_maniskill`, with the remainder timing out rather than misbehaving.

Taking over mid-episode is the case that matters for HIL, and it is verified
separately: the expert replans against the state the policy actually left it in,
including resuming at pre-insert rather than at the reach when the peg is
already in hand — restarting the reach would open the gripper and drop it. On
`insert_maniskill` one caveat is worth knowing: the plan needs 65–100 of the
episode's 100 steps, so a late takeover runs out of clock. Over twelve seeded
takeovers at k ∈ {5,…,30} random steps, 12/12 are rescued given a longer episode
but only 4/12 at the shipped limit, and every miss is a timeout at exactly
`100 - k` steps rather than a wrong action. Raise
`environment.config.max_episode_length` if you want late corrections to be able
to finish.

#### When the expert intervenes

Taking over on every step would just be behavior cloning on scripted data. Three
`hil.trigger` modes:

| mode | intervenes when | used for |
| --- | --- | --- |
| `always` | every step | bootstrapping demos with `record_demo` |
| `manual` | `‖a_expert‖ > hil.manual_deadband` (default `1e-3`), i.e. a human is pushing the device or holding a key | SpaceMouse, keyboard |
| `disagreement` | `‖a_policy − a_expert‖ > hil.disagreement_threshold` | scripted |

`disagreement` is shaped to resemble a human by two limits: `min_takeover_steps`
(nobody lets go after one step) and `max_intervention_ratio`, which decays
linearly to zero over `intervention_decay_steps` so the policy is progressively
left to fail on its own.

`hil.disagreement_threshold` must be calibrated per task, and both failure modes
are silent: below the smallest gap the policy and expert ever exhibit, the expert
drives every step and this is just behavior cloning; above the largest, it never
helps and this is plain SERL. The shipped values sit at the median of the gap
distribution measured against a random policy — `1.95` for the 6-dim Cartesian
task, `2.32` for the 8-dim joint-space one.

#### Human teleoperation in simulation

The scripted expert exists so the loop is reproducible without hardware. To be
the expert yourself there are two tasks. `insert_sim` is the 6-DoF Cartesian
delta `[dx, dy, dz, drx, dry, drz]` that both a SpaceMouse and the arrow-key
layout below map onto one-to-one. `pick_cube_sim` adds a seventh gripper
dimension and is SpaceMouse-only (see below). (`insert_maniskill` is 8-dim
joint-space, so neither device fits it without an IK layer.)

```bash
# keyboard, no hardware needed
./scripts/run_hil_serl.sh insert_sim_human actor

# SpaceMouse
EXPERT=spacemouse ./scripts/run_hil_serl.sh insert_sim_human actor
```

`config/task/insert_sim_human.yaml` selects `trigger: manual`, which is what
makes intervention hold-to-drive: while you are pushing the device or holding a
key the expert's action is non-zero and the wrapper hands over; the moment you
let go it emits zeros, the norm falls under `hil.manual_deadband`, and the
policy has control again. Nothing else changes — the same
`info["intervene_action"]` contract routes your correction into both buffers.

Note that `min_takeover_steps`, `max_intervention_ratio`, and
`intervention_decay_steps` are inert under `manual`; they exist to make the
*scripted* expert behave like a person, and here the person decides.

##### Keyboard layout

| key | action | key | action |
| --- | --- | --- | --- |
| `W` / `S` | ±x | `I` / `K` | ±rx |
| `A` / `D` | ±y | `J` / `L` | ±ry |
| `Q` / `E` | ±z | `U` / `O` | ±rz |
| `Shift` | fine mode (¼ scale) | `R` | request reset |

Two backends, chosen automatically. With [`pynput`](https://pypi.org/project/pynput/)
installed (`uv add pynput`) an action lasts exactly as long as the key is held.
Without it — or on a machine where pynput cannot reach the X server — the expert
falls back to reading the teleop window's `waitKey`, which reports key *repeats*
rather than holds; a press is then latched for `hil.expert_kwargs.sticky_steps`
control steps and fades out over them. That reads as jogging rather than
holding, which suits an insertion task made of small corrections, but expect to
tune `sticky_steps` to your own feel the first time.

##### The teleop window

`hil.hud: true` opens a cv2 window showing the wrist cameras alongside the
telemetry you need while driving: who currently has control (`HUMAN` / `POLICY`),
the action being executed, step and episode return, and the distance left to the
goal pose in millimetres and radians. That last row is the one to steer by —
`insert_sim` only pays out when every one of those six terms is inside
`reward_threshold`.

The window is skipped automatically when `DISPLAY`/`WAYLAND_DISPLAY` is unset,
so the same config runs unchanged on a headless training host; set `HUD=0` to
suppress it on a machine that does have a display. Note that the window is also
where the cv2 keyboard backend reads its keys, so closing it costs you keyboard
input unless pynput is installed.

##### Picking a cube with a SpaceMouse

`pick_cube_sim` is a grasp-and-carry task: the cube spawns at a random xy on the
table, a translucent green sphere marks a randomized 3D goal, and the reward pays
out only when the cube is **inside the goal and still held**. Requiring the grasp
is deliberate — otherwise flinging the cube through the sphere would score.

```bash
# SpaceMouse (the only supported device here — see below)
./scripts/run_hil_serl.sh pick_cube_sim_human actor

# record demonstrations by hand
DEMO_TRIGGER=manual ./scripts/run_hil_serl.sh pick_cube_sim_human demos 10
```

This is the only task with a **7-dim action**: the usual six Cartesian deltas
plus a gripper. The puck's **left button closes** and the **right button opens**.
The command is *latched*, not integrated — one click closes the fingers and they
stay closed until you click the other button. That is what lets a grasp survive
the whole carry while your hand is pushing translation only; an integrating
channel would drift open exactly then.

The keyboard expert is not a good fit here. It maps the gripper to `space` as a
*toggle*, and the cv2 backend cannot see holds, so driving a grasp and a carry at
once is fiddly. Use the SpaceMouse for this task.

Rough sequence, if you have not driven one before: hover above the cube, descend
until the fingers straddle it (~12 cm at the TCP), click left to close, wait a
beat for the contact to settle, then carry to the sphere.

`pick_cube_sim_human` raises the episode budget to 2000 steps and `time_limit` to
120 s — 40 s of simulated time, against the six seconds the base task allows. Both
have to move together because the first to fire ends the episode. This is far more
generous than `insert_sim_human`'s 1000 steps: a pick is four sub-tasks in series
and the grasp alone eats several seconds of hunting for the right height. If you
are still running out, raise both:

```bash
./scripts/run_hil_serl.sh pick_cube_sim_human demos 10 \
  --config_override=environment.config.max_episode_length=4000 \
  --config_override=environment.time_limit=240.0
```

##### Recording demonstrations by hand

`record_demo` drives the environment with zero actions and records whatever the
expert overrides them with, which is why the scripted flow forces
`trigger=always`. A human is not pushing on every step, so ask for `manual`
instead:

```bash
DEMO_TRIGGER=manual ./scripts/run_hil_serl.sh insert_sim_human demos 10
```

Idle steps are then recorded as the zero actions that genuinely executed. Only
successful episodes are kept.

##### Trying the device out first

`record_demo` is the wrong tool for finding out whether a device feels right: it
keeps only successful episodes, writes a multi-hundred-megabyte pickle at the
end, and tells you nothing about what the expert is actually emitting. Use the
tuning counterpart, which drives the same wrapper stack and **writes nothing**:

```bash
uv run python -m train.test_intervention --exp_name=pick_place_milk_human
```

The policy action is zero throughout, so everything the arm does is yours. Each
line shows what your device emitted (`exp`) next to what the environment executed
in the base frame (`base`), whether the wrapper counted the step as an
intervention (`intv`), and whether the object is grasped. Add `--episodes=N` to
stop after N attempts, or Ctrl-C at any point.

Two things it is good at catching: a device that jitters above
`manual_deadband` at rest shows a permanent `intv yes` while the arm sits still,
and a frame setting that did not take shows `exp` and `base` identical when you
expected them to differ.

##### Which frame the device drives in

`hil.expert_frame` decides whether your pushes are interpreted in the robot's
base frame or the gripper's own:

| value | a push "forward" moves the arm | good for |
| --- | --- | --- |
| `base` (default) | along the robot's fixed x axis | scripted experts, which compute deltas from world geometry |
| `tcp` | along whatever direction the gripper is pointing | hand-held devices |

`tcp` is what makes a SpaceMouse feel attached to the tool rather than to the
room, and it is set in `pick_place_milk_human`. It rotates only the translation
and rotation triplets — the gripper channel passes through, and the rotation is
skipped entirely when the task has no 7-dim `tcp_pose`, so the joint-space task
is unaffected. The trigger is judged on your raw device output, before the
rotation, so `manual_deadband` still means "how hard am I pushing".

Two details of `tcp`, both measured rather than reasoned about:

**Up stays up.** The rotation is yaw-only. Following the wrist's full
orientation would tie the lift axis to the tool's z, which points *downwards*
whenever the gripper is aimed at the table — pushing up would drive the arm into
the bin. Only the horizontal axes follow the wrist.

**`hil.expert_frame_yaw` squares the device with the flange.** A gripper's body
axes come from how the arm was assembled, not from anything you can see: the
Panda's tool x runs along world **+y** at this task's reset pose, so without a
correction a forward push moves the arm sideways on screen and a sideways push
moves it forward. That *swap* — rather than a mirror — is the giveaway.

Two symptoms, two different fixes, and telling them apart saves a lot of
guessing:

| what you see | what it means |
| --- | --- |
| forward↔sideways swapped | the yaw is 90° out |
| forward↔backward **and** left↔right reversed, up unchanged | the yaw is 180° out |

The shipped `90.0` is set from a SpaceMouse in hand, which is the only way to
settle it — the puck's physical axes are not derivable from the scene. If yours
disagrees, add or subtract 90 until it matches; the four values are the whole
space of possibilities.

If any of this feels wrong on your setup, `test_intervention` is the fastest way
to check: push one axis at a time and watch whether the arm moves the way you
expected.

##### Episode budget

`insert_sim` runs at 50 Hz, so the stock 100-step limit gives you two seconds of
simulated time per episode — far too short to hand-drive an insertion.
`insert_sim_human` therefore raises `max_episode_length` to 1000 and
`time_limit` to 60.0. Both matter: truncation fires on whichever arrives first.
The values are a starting point, not a measurement — adjust them to how fast you
actually work.

##### Warm-starting so the policy is not flailing when you take over

An untrained SAC policy is close to uniform noise, and two structural details of
this repo make that worse than it sounds for a human operator:

- **The sim actor outruns the learner about 11:1.** Unthrottled it steps at
  ~97 Hz while the learner manages ~8.6 updates/s, and weights are broadcast
  only every `steps_per_update` — about 5.8 s, by which point the actor has
  moved on ~565 steps. A correction you make is diluted across all of them. On
  real hardware physics does this pacing for you; in sim you have to ask.
- **Demonstrations do not reach the policy until online data does.** The
  learner's startup loop waits on the *replay* buffer, not the demo buffer, so
  the actor's first ~100 steps always run a randomly initialised network.

Two flags address these. `--actor_rate_hz` paces the rollout against the wall
clock (`pick_place_milk` sets 20 Hz in its `training:` block, matching the
env's own control rate), bringing the ratio to about 1:2.3. Budget for it: a
1375-step episode goes from ~11 s to ~69 s.

The setting lives in the **base** task rather than the `_human` variant on
purpose, because three loops step the same environment and all three have to
agree: the training actor, `record_demo`, and `test_intervention`. They share
one implementation (`infra/utils/rate_limit.py`) and all read
`training.actor_rate_hz`. A mismatch is invisible while it happens and costly
afterwards — an unpaced `test_intervention` runs at ~79 Hz with the HUD on and
~127 Hz without, so you would tune the SpaceMouse gains against an arm moving
four times faster than the one you then train with, and an unpaced
`record_demo` would write that same mismatch into the demo file the learner
samples half of every batch from. Pacing never changes the physics — a
simulator step advances a fixed slice of simulated time regardless — only how
often the trajectory is sampled. Both tools accept `--rate_hz` to override the
task setting; evaluation is left unpaced, since no human is in that loop.

`--pretrain_steps` behavior-clones the demo buffer before any online data
arrives and writes a checkpoint, which both processes then load on the next
launch:

```bash
# once, to produce the warm-start checkpoint
uv run python -m train.train_serl --exp_name=pick_place_milk_human \
  --demo_path=demo_data/pick_place_milk_human_demos.pkl \
  --checkpoint_path=checkpoints/pick_place_milk_human \
  --pretrain_steps=600 --pretrain_only --learner

# then learner and actor as usual, against that same --checkpoint_path
```

Measured over 150 steps against 20 hand-recorded episodes:

| | mean\|a\| | step-to-step jitter | gripper past the ±0.5 latch |
| --- | --- | --- | --- |
| random init | 0.650 | 0.678 | 53% |
| BC, mean only | 0.525 | 0.686 | 69% |
| **BC, mean + std** | **0.078** | **0.108** | **0%** |

That middle row is the trap, and it is why `update_bc` fits both halves of the
distribution: regressing only the mean takes `mean|mode|` from 0.43 to 0.03 —
the deterministic action is essentially fixed — while `scale` stays at its
initialised 0.2–2.3, so every *sampled* action still saturates tanh and the arm
still thrashes. From the outside the pretraining looks like it did nothing.

Note also that lowering the exploration noise on its own cannot fix this:
sweeping `std_max` from 5.0 to 0.1 moves `P(|a| > 0.9)` from 26% to 1.6% but
leaves `mean|mode|` at 0.413 the whole way. Only a gradient on the mean layer
moves the arm's actual target.

##### What the warm start does not do

The warm start is a starting point, not a safety net. Handed to SAC, it comes
apart within a couple of hundred online updates — measured, from scale 0.10 /
`|loc|` 0.03 to scale 0.76 / `|loc|` 0.84:

The cause is the sparse reward. Almost every demonstrated transition carries
reward 0, so the critic learns a Q that is nearly flat in the action — measured
−3.730 for the policy's own action, −3.731 for the demonstrated one, and −3.710
for a large random one. Maximising something that flat walks the policy straight
to the action bounds. Pretraining the critic for longer does not help; the
signal is not in the data to be learned.

This is not a bug to fix, and it is worth being clear about why: the HIL-SERL
paper has **no** warm start and **no** BC regularisation on the online actor
loss. Demonstrations only populate the demo buffer (§3.5). BC-then-RL is what
the paper's *baselines* do, and it argues explicitly against regularising toward
demonstrations — DAPG "performs similarly to the BC policies" and so
underperforms on tasks needing reactive behaviour (§4.5). Staying unconstrained
is how the policy is meant to exceed the demonstrator.

The paper's answer to early flailing is the operator. Its intervention rate
*starts* at 0.4–0.9 of all timesteps and decays to 0 over 1–2.5 h (Fig. 4). So
treat `--pretrain_steps` as a convenience that makes the first few minutes
easier to take over from, not as something that will hold on its own.

One sharp edge: `--pretrain_steps` counts towards the step number, so a resumed
run needs `training.max_steps` above it. The learner used to sit silently doing
nothing in that case (`range(start_step, max_steps)` is simply empty); it now
raises instead.

##### How to intervene

From the paper (§3.4, §3.5), and it is more specific than "help when it
struggles":

- **Correct, then let go.** "Issue specific corrections while letting the robot
  explore on its own otherwise." Take over when the policy reaches an
  unrecoverable state or is stuck, and release as soon as it is out.
- **Expect to drive a lot at first.** An early intervention rate of 50–90% is
  normal. There is no phase where the policy settles down by itself.
- **Do not drive all the way to success, repeatedly.** The paper warns that
  "persistently providing long sparse interventions that lead to task successes
  … will cause the overestimation of the value function, particularly in the
  early stages" and destabilise training.
- **Watch the intervention rate.** It should decay. If it does not, the policy
  is not improving — that is the failure signature the paper attributes to
  HG-DAgger.

#### Real hardware

On a machine with a SpaceMouse connected, switch experts without editing files:

```bash
EXPERT=spacemouse TRIGGER=manual ./scripts/run_hil_serl.sh insert_sim actor
```

`manual` reproduces the semantics of the existing `SpacemouseIntervention`
wrapper that `insert_real` uses. **This branch has not been tested against real
hardware** — it was developed in a container with no USB subsystem. The scripted
and SpaceMouse paths share all of their code beyond the expert itself. The
keyboard and HUD code paths were likewise verified only headless: the
degradation logic and the key-to-action mapping are tested, the on-screen window
itself is not.

#### Configuration

`config/task/insert_sim_hil.yaml` inherits the base task and adds only the
intervention block:

```yaml
defaults: [insert_sim, _self_]
hil:
  enabled: true
  expert: scripted_insert_sim
  trigger: disagreement
  disagreement_threshold: 1.95
  min_takeover_steps: 5
  max_intervention_ratio: 0.4
  intervention_decay_steps: 20000
```

Any key is overridable at launch, e.g.
`--config_override=hil.max_intervention_ratio=0.2`. Actor logs report
`intervention_count` and `intervention_steps` per episode — these are attached
to `info["episode"]` and reported through Weights & Biases, so they are not
visible when wandb is disabled.

Two things the script does for you. `train_serl` always initializes wandb, and
wandb *aborts the process* when no API key is configured — the learner would die
before its first step while the actor sat retrying a connection that never
comes. So the script passes `--debug` (wandb disabled) unless `WANDB_API_KEY` or
`~/.netrc` exists; set `DEBUG=0` to force real logging. And because the task
configs checkpoint every 5000 steps, `all` lowers `checkpoint_period` to half
the smoke budget — otherwise the eval stage aborts with "no checkpoint found".

Compared to upstream HIL-SERL this stack has no reward classifier, no BC or
gradient-penalty regularization, and no pre-training phase — only the
intervention loop.
