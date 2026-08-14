"""Play a recorded demo file back and check it is well formed.

    uv run python -m train.verify_demo --demo_path=demo_data/<file>.pkl

Two things happen at once, and the split matters.  The **checks** are
arithmetic over the file and always run, including headless -- they are what
tells you the data is usable.  The **playback** is a cv2 window showing the
frames that were recorded, and is only there so a person can see what the
operator actually did.  Pass ``--no_render`` (or run with no display) to keep
the checks and drop the window.

Playback replays *stored frames*; it does not re-drive the simulator.  That is
forced by how the data was made: ``record_demo``'s per-episode ``env.reset()``
takes no seed, so object placement is re-randomised and never recorded.
Feeding the stored actions back into a fresh environment would move the arm
through a scene whose carton is somewhere else -- it would look like a
successful demo replaying against the wrong world, which is worse than not
replaying at all, because the failure is silent and easy to misread as a bug in
the data.  The recorded images are the ground truth of what the operator saw.

What the checks look for, and why each one is a failure mode that has a cause
worth knowing:

- **Structure**: every transition carries the same keys and shapes.  A file
  that mixes shapes usually means two different task configs were recorded into
  one session.
- **Episode boundaries**: ``dones`` must mark the end of every episode, and
  ``record_demo`` keeps successes only, so every episode must also end with
  ``infos.succeed``.  An episode ending without it means truncation leaked into
  a file that is supposed to hold successes.
- **Episode length against the task budget**: an episode at exactly
  ``max_episode_length`` was truncated rather than completed.  Since the file
  only holds successes, that combination should not occur; if it does, the
  budget is too tight for the operator and later attempts are being discarded.
- **Reward**: this task is sparse -- reward is 1.0 exactly once per episode, on
  the final transition, and 0 everywhere else.
- **Actions**: within the action space, and not saturating it.  Actions pinned
  at the bounds mean the teleop gains are clipping the operator, who then
  cannot drive any faster no matter how hard they push the device.
- **Zero actions**: reported, not judged.  With ``trigger: manual`` an idle hand
  genuinely records zeros, so a moderate share is expected; a very high one
  means the deadband is eating real input.
"""

import pickle
from collections import Counter

import numpy as np
from absl import app, flags

from workspace import SERLWorkspace

FLAGS = flags.FLAGS
flags.DEFINE_string("demo_path", None, "Path to a .pkl written by record_demo.")
flags.DEFINE_string(
    "exp_name",
    None,
    "Task config the file was recorded with, used to check episode lengths "
    "against max_episode_length and actions against the action space. "
    "Inferred from the file name when omitted.",
)
flags.DEFINE_boolean("no_render", False, "Run the checks without a window.")
flags.DEFINE_float("fps", 20.0, "Playback rate. 0 plays as fast as it can.")
flags.DEFINE_integer("episode", -1, "Play only this episode index; -1 plays all.")
flags.DEFINE_multi_string("config_override", [], "Hydra task overrides.")

_OK = "\033[92m"
_BAD = "\033[91m"
_WARN = "\033[93m"
_DIM = "\033[2m"
_END = "\033[0m"


class _Report:
    """Collects pass/fail lines so one bad check does not hide the rest."""

    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, message):
        print(f"  {_OK}PASS{_END}  {message}")

    def fail(self, message):
        self.failures += 1
        print(f"  {_BAD}FAIL{_END}  {message}")

    def warn(self, message):
        self.warnings += 1
        print(f"  {_WARN}WARN{_END}  {message}")

    def note(self, message):
        print(f"  {_DIM}····{_END}  {message}")


def _episode_bounds(transitions):
    """Start/end index pairs, split on `dones`.

    A trailing episode with no final `done` is returned too -- reporting it as
    an episode is what lets the caller flag it, rather than dropping data
    silently.
    """
    bounds, start = [], 0
    for i, transition in enumerate(transitions):
        if bool(transition["dones"]):
            bounds.append((start, i))
            start = i + 1
    if start < len(transitions):
        bounds.append((start, len(transitions) - 1))
    return bounds


def _infer_exp_name(demo_path):
    """Recover the task from record_demo's `<exp_name>_<n>_demos_<stamp>.pkl`."""
    stem = demo_path.split("/")[-1]
    marker = "_demos"
    if marker not in stem:
        return None
    head = stem.split(marker)[0]
    # Strip the trailing `_<count>` that record_demo inserts.
    parts = head.rsplit("_", 1)
    candidate = parts[0] if len(parts) == 2 and parts[1].isdigit() else head
    return candidate if candidate in SERLWorkspace.available_tasks() else None


def _check_structure(transitions, report):
    keys = Counter(tuple(sorted(t.keys())) for t in transitions)
    if len(keys) == 1:
        report.ok(f"every transition carries the same keys ({len(transitions)} total)")
    else:
        report.fail(f"transitions disagree on keys: {list(keys)}")

    shapes = Counter(np.asarray(t["actions"]).shape for t in transitions)
    if len(shapes) == 1:
        report.ok(f"action shape consistent: {next(iter(shapes))}")
    else:
        report.fail(f"mixed action shapes: {dict(shapes)}")

    image_keys = sorted(
        k for k, v in transitions[0]["observations"].items() if k != "state"
    )
    report.note(
        f"observation: state{np.asarray(transitions[0]['observations']['state']).shape}"
        f", images {image_keys}"
    )
    return image_keys


def _check_episodes(transitions, bounds, max_episode_length, report):
    lengths = [end - start + 1 for start, end in bounds]
    report.note(
        f"{len(bounds)} episodes, lengths min={min(lengths)} "
        f"median={int(np.median(lengths))} max={max(lengths)}"
    )

    if bool(transitions[-1]["dones"]):
        report.ok("file ends on an episode boundary")
    else:
        report.fail(
            f"last episode has no terminal `done` -- {lengths[-1]} transitions "
            "were recorded after the previous episode ended, so the file was "
            "likely truncated mid-episode"
        )

    unsuccessful = [
        i for i, (_, end) in enumerate(bounds)
        if not bool(transitions[end]["infos"].get("succeed", False))
    ]
    if not unsuccessful:
        report.ok(f"every episode ends with succeed=True ({len(bounds)}/{len(bounds)})")
    else:
        report.fail(
            f"{len(unsuccessful)} episode(s) do not end in success: {unsuccessful}. "
            "record_demo keeps successes only, so these should not be here"
        )

    if max_episode_length is None:
        report.note("no task config resolved, skipping the episode-budget check")
        return
    at_budget = [i for i, n in enumerate(lengths) if n >= max_episode_length]
    if not at_budget:
        report.ok(
            f"every episode finished inside the budget "
            f"(longest {max(lengths)} < max_episode_length {max_episode_length})"
        )
    else:
        report.fail(
            f"episode(s) {at_budget} hit max_episode_length ({max_episode_length}), "
            "so they were truncated rather than completed"
        )


def _check_rewards(transitions, bounds, report):
    rewards = np.array([float(t["rewards"]) for t in transitions])
    nonzero = np.flatnonzero(rewards)
    ends = {end for _, end in bounds}
    misplaced = [int(i) for i in nonzero if int(i) not in ends]
    if misplaced:
        report.fail(
            f"{len(misplaced)} reward(s) outside a final transition, first at "
            f"index {misplaced[0]} -- this task is sparse and should pay only "
            "on success"
        )
    elif len(nonzero) == len(bounds):
        report.ok(
            f"sparse reward paid once per episode ({len(nonzero)}/{len(bounds)}), "
            f"values {sorted(set(rewards[nonzero].tolist()))}"
        )
    else:
        report.fail(
            f"{len(nonzero)} rewarded transitions for {len(bounds)} episodes"
        )

    masks = np.array([float(t["masks"]) for t in transitions])
    terminal_masks = np.array([masks[end] for _, end in bounds])
    if np.all(terminal_masks == 0.0):
        report.ok("terminal transitions carry mask 0 (bootstrapping stops there)")
    else:
        # Truncation is a time limit, not an absorbing state, so mask 1 at the
        # end is correct there -- worth surfacing but not a failure.
        report.warn(
            f"{int((terminal_masks != 0).sum())} episode(s) end with mask 1.0, "
            "which is right for truncation but not for a real terminal state"
        )


def _check_actions(transitions, action_space, bounds, report):
    actions = np.array([np.asarray(t["actions"], dtype=np.float64) for t in transitions])
    if not np.all(np.isfinite(actions)):
        report.fail("actions contain NaN or inf")
        return
    report.ok("all actions finite")

    if action_space is not None:
        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        if actions.shape[1] != low.shape[0]:
            report.fail(
                f"action width {actions.shape[1]} != env action space "
                f"{low.shape[0]} -- recorded with a different task config?"
            )
            return
        out = np.flatnonzero(
            (actions < low - 1e-6).any(1) | (actions > high + 1e-6).any(1)
        )
        if out.size:
            report.fail(f"{out.size} action(s) outside the action space")
        else:
            report.ok("every action lies inside the action space")

    # The Cartesian channels only; the gripper is a latch driven to +-1 on
    # purpose, so counting it as "saturated" would be meaningless.
    cartesian = actions[:, :6]
    saturated = np.abs(np.abs(cartesian) - 1.0) < 1e-6
    share = float(saturated.any(1).mean())
    peak = np.abs(cartesian).max(0)
    report.note(
        "per-axis peak |action|: "
        + " ".join(f"{v:.3f}" for v in peak)
        + f"   (translation x3, rotation x3)"
    )
    if share > 0.02:
        report.warn(
            f"{share:.1%} of steps saturate a Cartesian axis -- the teleop gains "
            "are clipping the operator, who cannot drive faster than this"
        )
    else:
        report.ok(f"Cartesian axes not saturating ({share:.2%} of steps at a bound)")

    idle = float((np.abs(actions).sum(1) == 0.0).mean())
    report.note(
        f"{idle:.1%} of steps are exactly zero "
        "(expected under trigger=manual: an idle hand records zeros)"
    )

    if actions.shape[1] >= 7:
        gripper = actions[:, 6]
        closes = int((gripper < -0.5).sum())
        opens = int((gripper > 0.5).sum())
        report.note(
            f"gripper latch: {closes} close commands, {opens} open commands, "
            f"{len(gripper) - closes - opens} holds"
        )
        # Judged per episode rather than over the file: a pick demo that never
        # closes the gripper cannot have succeeded, but the interesting failure
        # is one bad episode among good ones, which a file-wide count hides.
        never_close = [
            i for i, (start, end) in enumerate(bounds)
            if not (actions[start : end + 1, 6] < -0.5).any()
        ]
        if never_close:
            report.fail(
                f"episode(s) {never_close} never close the gripper "
                "(action[6] < -0.5), which a pick task cannot succeed without"
            )
        else:
            report.ok(f"every episode closes the gripper ({len(bounds)}/{len(bounds)})")


def _play(transitions, bounds, image_keys, report):
    import cv2

    from infra.utils.teleop_display import DISPLAY_AVAILABLE

    if not DISPLAY_AVAILABLE:
        report.note("no DISPLAY, skipping playback (checks above still ran)")
        return

    delay = max(1, int(round(1000.0 / FLAGS.fps))) if FLAGS.fps > 0 else 1
    window = "demo playback  --  q quits, space pauses"
    chosen = (
        list(enumerate(bounds))
        if FLAGS.episode < 0
        else [(FLAGS.episode, bounds[FLAGS.episode])]
    )

    for episode, (start, end) in chosen:
        for i in range(start, end + 1):
            transition = transitions[i]
            tiles = []
            for key in image_keys:
                frame = np.asarray(transition["observations"][key])
                # Stored with a leading obs_horizon axis; show the latest frame.
                if frame.ndim == 4:
                    frame = frame[-1]
                tiles.append(frame)
            canvas = np.hstack(tiles) if tiles else None
            if canvas is None:
                report.note("no images in this file, nothing to play")
                return

            canvas = np.ascontiguousarray(canvas[..., ::-1])  # RGB -> BGR for cv2
            scale = 3
            canvas = cv2.resize(
                canvas,
                (canvas.shape[1] * scale, canvas.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST,
            )
            action = np.asarray(transition["actions"], dtype=np.float64)
            grasped = bool(transition["infos"].get("grasped", False))
            label = (
                f"ep {episode}  step {i - start + 1}/{end - start + 1}  "
                f"grip {'closed' if grasped else 'open'}"
            )
            cv2.putText(
                canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (240, 240, 240), 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "a " + " ".join(f"{v:+.2f}" for v in action),
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
                cv2.LINE_AA,
            )
            if float(transition["rewards"]) > 0:
                cv2.putText(
                    canvas, "SUCCESS", (8, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 80), 2, cv2.LINE_AA,
                )
            cv2.imshow(window, canvas)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                report.note("playback stopped early")
                return
            if key == ord(" "):
                while (cv2.waitKey(50) & 0xFF) != ord(" "):
                    pass

    cv2.destroyAllWindows()


def main(_):
    if not FLAGS.demo_path:
        raise ValueError("--demo_path is required.")

    print(f"\nreading {FLAGS.demo_path}")
    with open(FLAGS.demo_path, "rb") as handle:
        transitions = pickle.load(handle)
    if not isinstance(transitions, list) or not transitions:
        raise ValueError(
            f"Expected a non-empty list of transitions, got "
            f"{type(transitions).__name__}."
        )

    exp_name = FLAGS.exp_name or _infer_exp_name(FLAGS.demo_path)
    max_episode_length = None
    action_space = None
    if exp_name:
        workspace = SERLWorkspace(exp_name, FLAGS.config_override)
        max_episode_length = workspace.raw_config.environment.config.get(
            "max_episode_length"
        )
        # fake_env: the spaces are all this needs, and building the real
        # simulator here would cost seconds and a GL context for nothing.
        action_space = workspace.get_environment(fake_env=True).action_space
        print(f"task {exp_name}  (max_episode_length={max_episode_length})")
    else:
        print(
            "no task config resolved from the file name; pass --exp_name to "
            "enable the episode-budget and action-space checks"
        )

    report = _Report()
    bounds = _episode_bounds(transitions)

    print("\nstructure")
    image_keys = _check_structure(transitions, report)
    print("\nepisodes")
    _check_episodes(transitions, bounds, max_episode_length, report)
    print("\nreward")
    _check_rewards(transitions, bounds, report)
    print("\nactions")
    _check_actions(transitions, action_space, bounds, report)

    print()
    if report.failures:
        print(f"{_BAD}{report.failures} check(s) failed{_END}", end="")
    else:
        print(f"{_OK}all checks passed{_END}", end="")
    if report.warnings:
        print(f", {_WARN}{report.warnings} warning(s){_END}")
    else:
        print()

    if not FLAGS.no_render:
        print("\nplaying back recorded frames  (q quits, space pauses)")
        _play(transitions, bounds, image_keys, report)

    # Non-zero exit on failure, so this is usable as a gate in a script.
    if report.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    app.run(main)
