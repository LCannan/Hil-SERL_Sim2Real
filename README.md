# SERL-Plus-Plus

![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

> This repository is built upon a fork of [HIL-SERL](https://github.com/rail-berkeley/hil-serl).

## Requirements

- Python 3.10.19
- CUDA 12.4+ recommended for GPU acceleration
- PyTorch 2.4.1+
- MuJoCo 3.4.0
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
uv run python -m train.train_rlpd \
  --exp_name=insert_sim \
  --demo_path=demo_data/peg_insert_sim_20_demos.pkl \
  --checkpoint_path=checkpoints/insert_sim \
  --learner

# Terminal 2: start actor node
uv run python -m train.train_rlpd \
  --exp_name=insert_sim \
  --checkpoint_path=checkpoints/insert_sim \
  --actor
```

### 2. Peg Insert Sim with Point Cloud

![peg_insert_pointcloud_sim](./doc/peg_insert_pointcloud_sim.gif)

```bash
# Download demo data
mkdir -p demo_data
wget -P demo_data https://github.com/liusong-0086/serl-plus-plus/releases/download/demo_data/peg_insert_pointcloud_sim_20_demos.pkl

# Terminal 1: start learner node
uv run python -m train.train_rlpd \
  --exp_name=insert_pointcloud_sim \
  --demo_path=demo_data/peg_insert_pointcloud_sim_20_demos.pkl \
  --checkpoint_path=checkpoints/insert_pointcloud_sim \
  --learner

# Terminal 2: start actor node
uv run python -m train.train_rlpd \
  --exp_name=insert_pointcloud_sim \
  --checkpoint_path=checkpoints/insert_pointcloud_sim \
  --actor
```

### 3. FR3 Real Robot

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
uv run python -m train.train_rlpd \
  --exp_name=insert_real \
  --demo_path=demo_data/<insert_real_demo_file>.pkl \
  --checkpoint_path=checkpoints/insert_real \
  --learner
```

Then start the actor on the computer connected to the robot, cameras, and
SpaceMouse:

```bash
uv run python -m train.train_rlpd \
  --exp_name=insert_real \
  --checkpoint_path=checkpoints/insert_real \
  --actor \
  --ip=<learner-ip>
```