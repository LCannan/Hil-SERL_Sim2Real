import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags

from infra.utils.rate_limit import RateLimiter, resolve_rate_hz
from algorithm.utils.gripper import (
    GRIPPER_STAY_INDEX,
    NUM_GRIPPER_ACTIONS,
    action_to_index,
)
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
flags.DEFINE_float(
    "rate_hz",
    0.0,
    "Cap the recording rate, in environment steps per wall-clock second. "
    "0 means take it from training.actor_rate_hz. This has to match the rate "
    "the actor runs at: the same hand motion sampled at two different rates is "
    "two different action distributions over identical physics, and the learner "
    "samples 50/50 from the demo buffer and the replay buffer.",
)


def _check_gripper_classes(transitions):
    """Warn if the recorded gripper column has lost its `stay` class.

    This data goes bad silently.  The discrete grasp critic sorts `action[-1]`
    into close/stay/open, and a healthy recording is ~98% stay -- the operator
    clicks twice per episode and the latch holds in between.  A wrapper that
    re-asserts the latch into the *recording* instead of only into the
    environment produces 0% stay, and nothing downstream raises: the class
    weighting simply learns that holding never happens, the head flips the
    gripper every step, and the arm never carries anything.  That cost a full
    training run here before it was spotted in wandb rather than at record time.
    """
    gripper = np.array(
        [np.asarray(t["actions"], dtype=np.float32)[-1] for t in transitions]
    )
    if gripper.size == 0:
        return
    counts = np.bincount(action_to_index(gripper), minlength=NUM_GRIPPER_ACTIONS)
    total = counts.sum()
    stay_frac = counts[GRIPPER_STAY_INDEX] / total
    print(
        f"gripper classes: close={counts[0]} ({counts[0]/total:.2%}) "
        f"stay={counts[1]} ({stay_frac:.2%}) open={counts[2]} ({counts[2]/total:.2%})"
    )
    if stay_frac >= 0.80:
        return
    print(
        "\n"
        "!! WARNING: only {:.1%} of recorded gripper commands are `stay`.\n"
        "!! A hand-recorded demo should be ~98% stay -- the latch is clicked\n"
        "!! twice an episode and held in between.  A low value means the held\n"
        "!! latch is being written into the recording rather than only into the\n"
        "!! environment, which trains the grasp critic to flip the gripper every\n"
        "!! step.  Do not train on this file; check ExpertIntervention.\n".format(
            stay_frac
        )
    )


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
    if np.asarray(transitions[0]["actions"]).shape[-1] > 6:
        _check_gripper_classes(transitions)
    return file_name


def main(_):
    workspace = SERLWorkspace(FLAGS.exp_name, FLAGS.config_override)
    env = workspace.get_environment(fake_env=False, seed=FLAGS.seed)

    # Pace the recording to whatever the actor will run at.  A scripted expert
    # does not care, but the resulting file does: demos recorded unthrottled at
    # ~127 Hz carry roughly a quarter of the per-step displacement of an
    # intervention made at 20 Hz, and both land in the same demo buffer.
    limiter = RateLimiter(resolve_rate_hz(workspace, FLAGS.rate_hz))
    if limiter.enabled:
        print(f"recording paced at {limiter.rate_hz:g} Hz")

    obs, info = env.reset(seed=FLAGS.seed)
    print("Reset done")
    limiter.reset()
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
                # A reset, and the deepcopy of a successful trajectory above,
                # both block for far longer than a step.
                limiter.reset()
            else:
                limiter.sleep()
    except KeyboardInterrupt:
        # Without this, Ctrl-C throws away every episode collected so far.
        print(f"\nInterrupted after {success_count} successful demos; saving.")

    _save(transitions, FLAGS.exp_name, success_count)
    env.close()

if __name__ == "__main__":
    app.run(main)
