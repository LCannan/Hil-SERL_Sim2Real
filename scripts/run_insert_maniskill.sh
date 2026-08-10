#!/usr/bin/env bash
#
# Train and evaluate the ManiSkill PegInsertionSide task.
#
#   ./scripts/run_insert_maniskill.sh prepare   # download dataset + build demos
#   ./scripts/run_insert_maniskill.sh learner   # terminal 1
#   ./scripts/run_insert_maniskill.sh actor     # terminal 2
#   ./scripts/run_insert_maniskill.sh eval [N]  # evaluate, writes N MP4s
#
# Evaluation renders off-screen and writes one MP4 per trajectory to
# $CHECKPOINT_PATH/videos -- no window is opened, matching the insert_sim flow.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

EXP_NAME="${EXP_NAME:-insert_maniskill}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/${EXP_NAME}}"
DATASET_DIR="${DATASET_DIR:-demo_data/rlt-maniskill-PegInsertionSide-v1-400-succ}"
DEMO_PATH="${DEMO_PATH:-demo_data/insert_maniskill_30_demos.pkl}"
NUM_DEMOS="${NUM_DEMOS:-30}"
SEED="${SEED:-42}"
# This scene's human-render camera is named `render_camera`; the flag's `front`
# default does not exist here.  Set to "" to drop the main view and keep only
# the wrist row.
EVAL_MAIN_CAMERA="${EVAL_MAIN_CAMERA:-render_camera}"

# Resolve uv even when PATH is minimal (cron, non-login shells, IDE terminals).
UV="${UV:-$(command -v uv || true)}"
if [[ -z "${UV}" ]]; then
    for candidate in "${HOME}/.local/bin/uv" /usr/local/bin/uv; do
        [[ -x "${candidate}" ]] && UV="${candidate}" && break
    done
fi
if [[ -z "${UV}" ]]; then
    echo "uv not found. Install it, or set UV=/path/to/uv." >&2
    exit 1
fi

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit "${1:-1}"
}

prepare() {
    if [[ ! -d "${DATASET_DIR}" ]]; then
        echo ">>> Downloading dataset (~9.2 GB) to ${DATASET_DIR}"
        "${UV}" run hf download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
            --repo-type dataset --local-dir "${DATASET_DIR}"
    else
        echo ">>> Dataset already present at ${DATASET_DIR}"
    fi

    if [[ ! -f "${DEMO_PATH}" ]]; then
        echo ">>> Converting ${NUM_DEMOS} episodes into ${DEMO_PATH}"
        "${UV}" run python train/convert_lerobot_demo.py \
            --dataset_path="${DATASET_DIR}" \
            --output_path="${DEMO_PATH}" \
            --num_episodes="${NUM_DEMOS}" \
            --seed="${SEED}"
    else
        echo ">>> Demos already present at ${DEMO_PATH}; validating"
        "${UV}" run python train/convert_lerobot_demo.py \
            --validate_only --output_path="${DEMO_PATH}"
    fi
}

learner() {
    [[ -f "${DEMO_PATH}" ]] || {
        echo "Missing ${DEMO_PATH}. Run '$0 prepare' first." >&2
        exit 1
    }
    mkdir -p "${CHECKPOINT_PATH}"
    exec "${UV}" run python -m train.train_serl \
        --exp_name="${EXP_NAME}" \
        --demo_path="${DEMO_PATH}" \
        --checkpoint_path="${CHECKPOINT_PATH}" \
        --seed="${SEED}" \
        --learner "$@"
}

actor() {
    mkdir -p "${CHECKPOINT_PATH}"
    exec "${UV}" run python -m train.train_serl \
        --exp_name="${EXP_NAME}" \
        --checkpoint_path="${CHECKPOINT_PATH}" \
        --seed="${SEED}" \
        --eval_video_main_camera="${EVAL_MAIN_CAMERA}" \
        --actor "$@"
}

evaluate() {
    local n_trajs="${1:-10}"
    shift || true
    exec "${UV}" run python -m train.train_serl \
        --exp_name="${EXP_NAME}" \
        --checkpoint_path="${CHECKPOINT_PATH}" \
        --seed="${SEED}" \
        --eval_n_trajs="${n_trajs}" \
        --eval_video_main_camera="${EVAL_MAIN_CAMERA}" \
        --actor "$@"
}

command="${1:-}"
shift || true
case "${command}" in
    prepare) prepare ;;
    learner) learner "$@" ;;
    actor) actor "$@" ;;
    eval|evaluate) evaluate "$@" ;;
    -h|--help|help) usage 0 ;;
    *) echo "Unknown command: ${command:-<none>}" >&2; usage 1 ;;
esac
