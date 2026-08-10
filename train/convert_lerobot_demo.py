"""Convert the LeRobot ManiSkill PegInsertionSide dataset into a SERL demo pkl.

Turns ``demo_data/rlt-maniskill-PegInsertionSide-v1-400-succ`` (LeRobot v2.1)
into the list-of-transition-dicts format that ``train_serl.py --demo_path``
loads, matching the ``insert_maniskill`` task's observation and action spaces.

Example:

    python train/convert_lerobot_demo.py \\
        --dataset_path demo_data/rlt-maniskill-PegInsertionSide-v1-400-succ \\
        --output_path demo_data/insert_maniskill_30_demos.pkl \\
        --num_episodes 30

Images are downsampled through the *same* ``resize_rgb`` the environment uses,
so demo frames and online frames are processed identically by construction.
"""

import json
import pickle as pkl
import random
from pathlib import Path

import numpy as np
from absl import app, flags
from PIL import Image
from tqdm import tqdm

# maniskill_peg_gym_env's module body is deliberately free of mani_skill/sapien
# imports, so this works without the optional extra installed.  Keep it that way
# rather than duplicating the resize here -- sharing the helper is what makes
# demo and online image processing provably identical.
from infra.sim.envs.maniskill_peg_gym_env import resize_rgb

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dataset_path",
    "demo_data/rlt-maniskill-PegInsertionSide-v1-400-succ",
    "Root of the LeRobot dataset (the directory containing meta/ and data/).",
)
flags.DEFINE_string(
    "output_path",
    "demo_data/insert_maniskill_30_demos.pkl",
    "Where to write the demo pkl.",
)
flags.DEFINE_integer("num_episodes", 30, "How many episodes to convert.")
flags.DEFINE_integer("seed", 42, "Seed for random episode selection.")
flags.DEFINE_enum(
    "episode_selection",
    "random",
    ["random", "first"],
    "Sample episodes randomly, or take the first num_episodes.",
)
flags.DEFINE_float(
    "sparse_reward_on_success",
    1.0,
    "Reward on the terminal transition. Must match the task config's "
    "environment.config.sparse_reward_on_success.",
)
flags.DEFINE_integer("image_size", 128, "Square size the images are resized to.")
flags.DEFINE_bool(
    "validate_only",
    False,
    "Skip conversion and only run the validation pass over an existing "
    "--output_path.",
)

_STATE_DIM = 9
_ACTION_DIM = 8
_TRANSITION_KEYS = {
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
}
# Must match infra/sim/envs/maniskill_peg_variants.py.
MAIN_IMAGE_KEY = "3rd_view_camera"
WRIST_IMAGE_KEY = "wide_hand_camera"
# LeRobot column -> SERL image key.
_IMAGE_COLUMNS = {"image": MAIN_IMAGE_KEY, "wrist_image": WRIST_IMAGE_KEY}


def _load_meta(dataset_path: Path):
    info_path = dataset_path / "meta" / "info.json"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset, missing {info_path}")

    with open(info_path) as f:
        info = json.load(f)
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    return info, episodes


def _episode_parquet_path(dataset_path: Path, info: dict, episode_index: int) -> Path:
    # Derived from the template rather than hardcoding chunk-000, so this keeps
    # working for datasets that spill past one chunk.
    chunk = episode_index // int(info["chunks_size"])
    relative = info["data_path"].format(
        episode_chunk=chunk, episode_index=episode_index
    )
    return dataset_path / relative


def _decode_image(value, size: int) -> np.ndarray:
    """Decode one LeRobot image cell to a resized uint8 RGB array."""
    if isinstance(value, dict):
        # HF Image structs arrive as {"bytes": ..., "path": ...}.
        value = value.get("bytes")
    if isinstance(value, (bytes, bytearray)):
        import io

        # PIL's convert("RGB") yields RGB directly; cv2.imdecode would give BGR.
        image = np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    else:
        image = np.asarray(value)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
        image = image.astype(np.uint8, copy=False)
    return resize_rgb(image, (size, size))


def _load_episode(dataset_path: Path, info: dict, episode: dict, size: int):
    import pyarrow.parquet as pq

    index = int(episode["episode_index"])
    path = _episode_parquet_path(dataset_path, info, index)
    if not path.is_file():
        raise FileNotFoundError(f"Missing parquet for episode {index}: {path}")

    table = pq.read_table(path).to_pydict()
    length = len(table["state"])
    if length != int(episode["length"]):
        raise ValueError(
            f"Episode {index}: parquet has {length} rows but episodes.jsonl "
            f"declares {episode['length']}. The download may be truncated."
        )

    # to_pydict() widens these to float64 despite info.json declaring float32;
    # cast explicitly or ReplayBuffer silently narrows them on insert.
    states = np.asarray(table["state"], dtype=np.float32)
    actions = np.asarray(table["actions"], dtype=np.float32)
    if states.shape[1:] != (_STATE_DIM,):
        raise ValueError(f"Episode {index}: expected 9-dim state, got {states.shape}")
    if actions.shape[1:] != (_ACTION_DIM,):
        raise ValueError(f"Episode {index}: expected 8-dim action, got {actions.shape}")

    images = {
        key: [_decode_image(cell, size) for cell in table[column]]
        for column, key in _IMAGE_COLUMNS.items()
    }
    return states, actions, images, length


def _build_transitions(states, actions, images, length, reward_on_success):
    # Build each observation once and share the reference between the transition
    # that owns it and the one before it.  pickle memoizes by identity, so every
    # frame lands in the file exactly once.
    observations = [
        {
            "state": states[i][None],
            MAIN_IMAGE_KEY: images[MAIN_IMAGE_KEY][i][None],
            WRIST_IMAGE_KEY: images[WRIST_IMAGE_KEY][i][None],
        }
        for i in range(length)
    ]

    transitions = []
    # length - 1 transitions: next_observations structurally cannot cross an
    # episode boundary.  Note the collector dropped the true terminal frame, so
    # the last observation here is the closest-to-insertion state available.
    for i in range(length - 1):
        terminal = i == length - 2
        transitions.append(
            {
                "observations": observations[i],
                "actions": actions[i],
                "next_observations": observations[i + 1],
                "rewards": np.asarray(
                    reward_on_success if terminal else 0.0, dtype=np.float32
                ),
                # masks = 1 - terminated, matching train_serl.py.
                "masks": np.asarray(0.0 if terminal else 1.0, dtype=np.float32),
                "dones": bool(terminal),
            }
        )
    return transitions


def validate(transitions, num_episodes: int, size: int, reward_on_success: float):
    """Assert the pkl is exactly what ReplayBufferDataStore expects."""
    if not transitions:
        raise ValueError("No transitions were produced")

    image_shape = (1, size, size, 3)
    done_count = 0
    for i, transition in enumerate(transitions):
        keys = set(transition)
        if keys != _TRANSITION_KEYS:
            raise ValueError(
                f"transition {i}: key set {sorted(keys)} != {sorted(_TRANSITION_KEYS)}"
            )

        for field in ("observations", "next_observations"):
            obs = transition[field]
            expected = {"state", MAIN_IMAGE_KEY, WRIST_IMAGE_KEY}
            if set(obs) != expected:
                raise ValueError(
                    f"transition {i} {field}: keys {sorted(obs)} != {sorted(expected)}"
                )
            state = obs["state"]
            # A (9,) state would be silently broadcast into the (1, 9) buffer
            # slot, so check the rank as well as the dtype.
            if state.shape != (1, _STATE_DIM) or state.dtype != np.float32:
                raise ValueError(
                    f"transition {i} {field}: state must be (1, {_STATE_DIM}) "
                    f"float32, got {state.shape} {state.dtype}"
                )
            for key in (MAIN_IMAGE_KEY, WRIST_IMAGE_KEY):
                image = obs[key]
                if image.shape != image_shape or image.dtype != np.uint8:
                    raise ValueError(
                        f"transition {i} {field}: {key} must be {image_shape} "
                        f"uint8, got {image.shape} {image.dtype}"
                    )

        action = transition["actions"]
        if action.shape != (_ACTION_DIM,) or action.dtype != np.float32:
            raise ValueError(
                f"transition {i}: actions must be ({_ACTION_DIM},) float32, got "
                f"{action.shape} {action.dtype}"
            )
        if np.abs(action).max() > 1.0 + 1e-6:
            raise ValueError(f"transition {i}: actions outside [-1, 1]")

        done = bool(transition["dones"])
        reward = float(transition["rewards"])
        mask = float(transition["masks"])
        if done != (reward > 0) or done != (mask == 0.0):
            raise ValueError(
                f"transition {i}: dones/rewards/masks disagree "
                f"(dones={done}, rewards={reward}, masks={mask})"
            )
        if done:
            done_count += 1
            if reward != reward_on_success:
                raise ValueError(
                    f"transition {i}: terminal reward {reward} != "
                    f"{reward_on_success}"
                )

        # Adjacent transitions must share the observation object, not copy it.
        if i and transitions[i - 1]["next_observations"] is not transition[
            "observations"
        ] and not transitions[i - 1]["dones"]:
            raise ValueError(
                f"transition {i}: observation is not shared with the previous "
                "transition; the pkl will be far larger than necessary"
            )

    if done_count != num_episodes:
        raise ValueError(
            f"expected {num_episodes} terminal transitions, found {done_count}"
        )

    sample = transitions[0]["observations"]
    for key in (MAIN_IMAGE_KEY, WRIST_IMAGE_KEY):
        image = sample[key]
        if image.max() == 0:
            raise ValueError(f"{key} is all zeros -- the render path failed")
        if image.std() == 0:
            raise ValueError(f"{key} is a constant image")

    print(
        f"validated {len(transitions)} transitions over {done_count} episodes: "
        f"state (1, {_STATE_DIM}) float32, images {image_shape} uint8, "
        f"actions ({_ACTION_DIM},) float32"
    )


def main(_):
    output_path = Path(FLAGS.output_path)
    size = int(FLAGS.image_size)

    if FLAGS.validate_only:
        with open(output_path, "rb") as f:
            transitions = pkl.load(f)
        validate(
            transitions,
            num_episodes=sum(bool(t["dones"]) for t in transitions),
            size=size,
            reward_on_success=FLAGS.sparse_reward_on_success,
        )
        return

    dataset_path = Path(FLAGS.dataset_path)
    info, episodes = _load_meta(dataset_path)
    if FLAGS.num_episodes > len(episodes):
        raise ValueError(
            f"--num_episodes={FLAGS.num_episodes} exceeds the {len(episodes)} "
            "episodes in the dataset"
        )

    if FLAGS.episode_selection == "random":
        selected = random.Random(FLAGS.seed).sample(episodes, FLAGS.num_episodes)
        selected.sort(key=lambda item: item["episode_index"])
    else:
        selected = episodes[: FLAGS.num_episodes]

    transitions = []
    for episode in tqdm(selected, desc="episodes"):
        states, actions, images, length = _load_episode(
            dataset_path, info, episode, size
        )
        transitions.extend(
            _build_transitions(
                states, actions, images, length, FLAGS.sparse_reward_on_success
            )
        )

    # Validate before writing, so a malformed pkl never reaches disk.
    validate(
        transitions,
        num_episodes=len(selected),
        size=size,
        reward_on_success=FLAGS.sparse_reward_on_success,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump(transitions, f)
    print(
        f"saved {len(transitions)} transitions from {len(selected)} episodes to "
        f"{output_path}"
    )


if __name__ == "__main__":
    app.run(main)
