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

### 4. FR3 Real Robot

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
hardware, the expert is an interface with two implementations — a SpaceMouse and
a scripted controller — selected by config:

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
| `manual` | `‖a_expert‖ > 1e-3`, i.e. the human is pushing the device | SpaceMouse |
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

#### Real hardware

On a machine with a SpaceMouse connected, switch experts without editing files:

```bash
EXPERT=spacemouse TRIGGER=manual ./scripts/run_hil_serl.sh insert_sim actor
```

`manual` reproduces the semantics of the existing `SpacemouseIntervention`
wrapper that `insert_real` uses. **This branch has not been tested against real
hardware** — it was developed in a container with no USB subsystem. The scripted
and SpaceMouse paths share all of their code beyond the expert itself.

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
