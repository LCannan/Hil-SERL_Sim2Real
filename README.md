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
runs on another machine, pass the learner address with `--ip=<learner-ip>` or
edit the corresponding run script.

### 1. Peg Insert Sim with RGB

![peg_insert_sim](./doc/peg_insert_sim.gif)

```bash
# Enter the experiment folder
cd demos/experiments/peg_insert_sim

# Download demo data
mkdir -p demo_data
cd demo_data
wget https://github.com/liusong-0086/serl-plus-plus/releases/download/demo_data/peg_insert_sim_20_demos.pkl
cd ..

# Terminal 1: start learner node
bash run_learner.sh

# Terminal 2: start actor node
bash run_actor.sh
```

### 2. Peg Insert Sim with Point Cloud

![peg_insert_pointcloud_sim](./doc/peg_insert_pointcloud_sim.gif)

```bash
# Enter the experiment folder
cd demos/experiments/peg_insert_pointcloud_sim

# Download demo data
mkdir -p demo_data
cd demo_data
wget https://github.com/liusong-0086/serl-plus-plus/releases/download/demo_data/peg_insert_pointcloud_sim_20_demos.pkl
cd ..

# Terminal 1: start learner node
bash run_learner.sh

# Terminal 2: start actor node
bash run_actor.sh
```