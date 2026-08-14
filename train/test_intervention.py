"""Try out human intervention without recording anything.

``record_demo.py`` is the wrong tool for finding out whether a device feels
right: it keeps only successful episodes, writes a multi-hundred-megabyte pickle
at the end, and gives no feedback about what the expert is actually emitting.
This is the tuning counterpart -- it drives the same wrapper stack the real runs
use, prints what the expert produced, and **never writes a file**.

    uv run python -m train.test_intervention --exp_name=pick_place_milk_human

The policy action is zero throughout, so whatever the arm does is the operator's
doing.  That mirrors ``record_demo``: with ``trigger: manual`` an idle hand
leaves the arm still, which is exactly the signal you want when checking whether
a deadband is set sensibly.

Two things worth watching in the live line, both of which have been real bugs
here rather than hypothetical ones:

- ``exp`` is what the operator's device emitted and ``base`` is what the
  environment actually executed, in the robot's base frame.  Under
  ``hil.expert_frame: tcp`` they differ by the wrist rotation; under ``base``
  they are identical.  If you expect a rotation and see none, the config did not
  take.  (``info["intervene_action"]`` is *not* shown: ``RelativeFrame`` rewrites
  it into the policy's frame on the way out, so it matches neither of these.)
- ``intv`` shows whether the wrapper counted the step as an intervention.  A
  device that jitters above ``manual_deadband`` at rest shows up here as a
  permanent ``yes`` with the arm not moving.
"""

import time

import numpy as np
from absl import app, flags

from infra.utils.rate_limit import RateLimiter, resolve_rate_hz
from workspace import SERLWorkspace

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Task config, usually a _human variant.")
flags.DEFINE_multi_string(
    "config_override", [], "Hydra task overrides, e.g. hil.expert_frame=base."
)
flags.DEFINE_integer("episodes", 0, "Episodes to run; 0 means until Ctrl-C.")
flags.DEFINE_integer("seed", 42, "Environment seed.")
flags.DEFINE_integer(
    "print_every",
    5,
    "Print the live line every N steps. Raise it if the output scrolls too fast.",
)
flags.DEFINE_float(
    "rate_hz",
    0.0,
    "Cap the rollout rate, in environment steps per wall-clock second. 0 means "
    "take it from training.actor_rate_hz. Leave it alone unless you are "
    "deliberately testing a different rate: the point of this tool is that the "
    "arm moves under your hand exactly as fast as it will during training.",
)


def _summarise(episode, steps, intervened, succeeded, elapsed):
    rate = 100.0 * intervened / steps if steps else 0.0
    outcome = "SUCCESS" if succeeded else "failed "
    print(
        f"  episode {episode}: {outcome}  {steps:4d} steps  "
        f"{rate:5.1f}% intervened  {elapsed:5.1f}s"
    )


def main(_):
    if not FLAGS.exp_name:
        choices = ", ".join(SERLWorkspace.available_tasks())
        raise ValueError(f"--exp_name is required. Available tasks: {choices}")

    workspace = SERLWorkspace(FLAGS.exp_name, FLAGS.config_override)
    hil = workspace.raw_config.get("hil")
    if hil is None or not hil.get("enabled"):
        raise ValueError(
            f"Task {FLAGS.exp_name!r} has no enabled `hil` block, so there is no "
            "expert to test. Pick a _human or _hil variant, or pass "
            "--config_override=hil.enabled=true."
        )

    print(
        f"expert={hil.get('expert')}  trigger={hil.get('trigger')}  "
        f"frame={hil.get('expert_frame', 'base')}  "
        f"deadband={hil.get('manual_deadband', 1e-3)}"
    )
    print("Nothing is recorded. Ctrl-C to stop.\n")

    env = workspace.get_environment(fake_env=False, seed=FLAGS.seed)

    # Pace to the rate the actor will run at.  Without this the loop runs flat
    # out (~79 Hz here with the HUD on, ~127 without, and varying with machine
    # load), so the arm moves several times faster under the operator's hand
    # than it will during training -- which makes every gain tuned here wrong
    # for the run it was tuned for.
    limiter = RateLimiter(resolve_rate_hz(workspace, FLAGS.rate_hz))
    print(
        f"paced at {limiter.rate_hz:g} Hz"
        if limiter.enabled
        else "UNPACED: this task sets no training.actor_rate_hz, so the arm "
        "will move faster here than during training."
    )
    # The wrapper stack is built outermost-first, so walk in to the one layer
    # that knows what the expert emitted before any frame change.
    intervention = env
    while intervention is not None and not hasattr(intervention, "expert"):
        intervention = getattr(intervention, "env", None)

    episode = 0
    try:
        while not FLAGS.episodes or episode < FLAGS.episodes:
            episode += 1
            obs, _ = env.reset()
            steps = intervened = 0
            succeeded = False
            started = time.monotonic()
            limiter.reset()

            while True:
                # Zero policy action: everything that happens is the operator's.
                obs, reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                steps += 1
                executed = info.get("intervene_action")
                if executed is not None:
                    intervened += 1
                succeeded = bool(info.get("succeed", False))

                if steps % FLAGS.print_every == 0:
                    expert_action = getattr(intervention, "_last_expert", None)
                    executed_base = getattr(intervention, "_last_executed", None)
                    fmt = lambda v: np.array2string(
                        np.asarray(v)[:3], precision=2, floatmode="fixed", sign="+"
                    )
                    parts = [f"ep{episode} step {steps:4d}"]
                    if expert_action is not None:
                        parts.append(f"exp {fmt(expert_action)}")
                    parts.append(
                        f"base {fmt(executed_base)}"
                        if executed_base is not None
                        else "base   (policy)   "
                    )
                    parts.append(f"intv {'yes' if executed_base is not None else ' no'}")
                    parts.append(f"grip {float(info.get('grasped', 0)):.0f}")
                    if succeeded:
                        parts.append("SUCCEED")
                    # Padded so a shorter line cannot leave debris from a longer
                    # one behind it on the same carriage return.
                    print("  " + "  ".join(parts).ljust(96), end="\r", flush=True)

                if terminated or truncated:
                    print(" " * 100, end="\r")
                    _summarise(
                        episode, steps, intervened, succeeded,
                        time.monotonic() - started,
                    )
                    break

                limiter.sleep()
    except KeyboardInterrupt:
        print("\nStopped. Nothing was written.")
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
