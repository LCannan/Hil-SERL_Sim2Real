"""Wall-clock pacing for rollout loops.

Shared by every entrypoint that steps an environment a human might be watching
or driving: ``train_serl``'s actor, ``record_demo``, and ``test_intervention``.
It lives here rather than in ``train_serl`` because the three loops have to run
at the *same* rate or the data does not match the operator's hands -- a demo
recorded at 127 Hz and an intervention made at 20 Hz are different action
distributions over the same physics, and ``next_training_batch`` samples 50/50
from both.

Only the wall clock is affected.  A simulator step advances a fixed slice of
simulated time whatever the rate, so pacing changes how a trajectory is
*sampled*, never what the physics computes.
"""

import time
from typing import Any


class RateLimiter:
    """Hold a loop to ``rate_hz`` environment steps per wall-clock second.

    Deadline-based rather than a flat ``sleep(period)``: sleeping a fixed
    period adds the loop's own cost on top of it, so the loop drifts slower
    than asked for -- and the drift grows with whatever the step happens to
    cost that iteration, which is exactly the jitter this class exists to
    remove.
    """

    def __init__(self, rate_hz: float):
        self.rate_hz = float(rate_hz or 0.0)
        if self.rate_hz < 0.0:
            raise ValueError(f"rate_hz must be non-negative, got {rate_hz}.")
        self.period = 1.0 / self.rate_hz if self.rate_hz else 0.0
        self._deadline = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self.period > 0.0

    def reset(self) -> None:
        """Restart the schedule from now.

        Call after anything that legitimately blocks for longer than a step --
        an ``env.reset``, a buffer dump -- so the pause is not treated as a
        backlog to be caught up on.
        """
        self._deadline = time.monotonic()

    def sleep(self) -> None:
        """Block until this step's deadline.  A no-op when unthrottled.

        Call it *after* the step's work and after any timer has been stopped,
        so the timers report work done rather than time waited.
        """
        if not self.enabled:
            return
        self._deadline += self.period
        remaining = self._deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            # Fell behind (a slow render, a checkpoint dump).  Give up on the
            # backlog rather than sprinting through it, which would defeat the
            # point of pacing an operator.
            self._deadline = time.monotonic()


def resolve_rate_hz(workspace: Any, flag_value: float = 0.0) -> float:
    """Pick the rollout rate: an explicit flag wins, else the task config.

    Read through ``getattr`` rather than as a required config key -- tasks that
    are never hand-driven and never feed a HIL run have no reason to carry the
    setting at all, and ``BaseWorkspace.__getattr__`` raises on a missing one.
    """
    if flag_value:
        return float(flag_value)
    return float(getattr(workspace, "actor_rate_hz", 0.0) or 0.0)
