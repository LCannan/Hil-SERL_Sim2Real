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

For the real robot, first edit camera, pose, safety-limit, and controller values
in `config/task/insert_real.yaml`, then use `--exp_name=insert_real`. For example,
`--config_override=training.batch_size=128` changes a task value without editing
the YAML file.