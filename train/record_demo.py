import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags

from workspace import SERLWorkspace

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_multi_string(
    "config_override",
    [],
    "Hydra task overrides, for example environment.config.random_reset=true.",
)
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer(
    "max_attempts",
    0,
    "Give up after this many episodes even if fewer demos were collected. "
    "0 means no limit. Useful with a scripted expert, whose success rate is "
    "below 1.0 -- without a cap a misconfigured expert loops forever.",
)
flags.DEFINE_integer("seed", 42, "Environment seed.")


def _save(transitions, exp_name, count):
    if not transitions:
        print("No successful episodes to save.")
        return None
    os.makedirs("./demo_data", exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{exp_name}_{count}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
    print(f"saved {count} demos ({len(transitions)} transitions) to {file_name}")
    return file_name


def main(_):
    workspace = SERLWorkspace(FLAGS.exp_name, FLAGS.config_override)
    env = workspace.get_environment(fake_env=False, seed=FLAGS.seed)

    obs, info = env.reset(seed=FLAGS.seed)
    print("Reset done")
    transitions = []
    success_count = 0
    attempts = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, terminated, truncated, info = env.step(actions)
            episode_done = terminated or truncated
            returns += rew

            if "total_action" in info:
                actions = info["total_action"]

            if "intervene_action" in info:
                actions = info["intervene_action"]
                actions = np.clip(actions, env.action_space.low,
                                           env.action_space.high)

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=np.asarray(1.0 - float(terminated), dtype=np.float32),
                    dones=episode_done,
                    infos=info,
                )
            )
            trajectory.append(transition)

            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if episode_done:
                attempts += 1
                if info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                trajectory = []
                returns = 0
                if FLAGS.max_attempts and attempts >= FLAGS.max_attempts:
                    print(
                        f"\nStopping after {attempts} attempts with "
                        f"{success_count}/{success_needed} successes."
                    )
                    break
                obs, info = env.reset()
    except KeyboardInterrupt:
        # Without this, Ctrl-C throws away every episode collected so far.
        print(f"\nInterrupted after {success_count} successful demos; saving.")

    _save(transitions, FLAGS.exp_name, success_count)
    env.close()

if __name__ == "__main__":
    app.run(main)
