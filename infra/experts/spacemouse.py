"""Adapter exposing the existing SpaceMouse driver through the Expert protocol.

Thin by design: :class:`~infra.hardware.spacemouse.spacemouse_expert.SpaceMouseExpert`
already has the right ``get_action``/``close`` signatures.  This wrapper adds
an actionable import error, a fixed 2-element button tuple, and the response
shaping (deadzone / expo / scale) that makes a 6-DoF puck usable for fine
insertion work.

Both buttons are **edge-triggered**, which is what makes them toggles rather
than the hold-to-act bindings they used to be:

- left  -- flips the gripper between closed and open.
- right -- flips the intervention mode; :class:`ExpertIntervention` reads the
  request and decides what the two modes mean.
"""

import math
from typing import Any, List, Tuple

import numpy as np

from .base import ExpertBase

LEGEND = "puck=move  L=grip toggle  R=mode toggle"


class SpaceMouseAdapter(ExpertBase):
    """A human driving a 3Dconnexion SpaceMouse.

    Requires a real device.  Inside a container with no USB subsystem, opening
    it fails with ``Exception("HID API is probably not installed")`` from deep
    inside ``pyspacemouse``; we translate that into something actionable.
    """

    def __init__(
        self,
        env: Any = None,
        action_dim: int = 6,
        seed: int = 0,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        deadzone: float = 0.0,
        expo: float = 1.0,
        invert_xy: bool = False,
        **kwargs: Any,
    ):
        # `seed` is accepted and ignored, matching the scripted experts: the
        # factory seeds every expert uniformly, and without this the guard
        # below rejects it and `hil.expert=spacemouse` cannot be built at all.
        del env, seed  # a human needs neither privileged state nor a seed
        self.action_dim = int(action_dim)
        self.gripper_scale = float(kwargs.pop("gripper_scale", 1.0))
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.deadzone = float(deadzone)
        self.expo = float(expo)
        # Negates the two horizontal translation axes.  This is a property of
        # where the *camera* is, not of the device: a scene camera facing the
        # robot mirrors the horizontal plane on screen, so pushing the puck
        # right sends the arm left in the picture the operator is watching.
        # Only the human's own actions are affected -- a scripted expert reads
        # simulator state directly and never passes through here.
        self.invert_xy = bool(invert_xy)
        # Gripper latch and mode request, both driven by button *edges*.  The
        # latch is held asserted on every step rather than pulsed once: the
        # environments latch on |action[6]| > 0.5 and would accept a pulse, but
        # in `manual` mode the policy drives the gripper channel on every step
        # the human is not intervening -- and with `grasp_critic` its greedy
        # argmax is exactly -1/0/+1, which clears that threshold every time.
        # A pulsed grasp would be released by the policy one step later.
        self._gripper_closed = False
        self._mode_toggle_requested = False
        # Zeros, not None: the reader publishes zeroed counters before this
        # object exists, so there is no baseline to adopt -- and adopting one on
        # the first call would swallow the session's first click.
        self._press_counts = [0, 0, 0, 0]
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError(
                f"deadzone must be in [0, 1), got {self.deadzone}. It is a "
                "fraction of the puck's full deflection, not a metric distance."
            )
        if self.expo <= 0.0:
            raise ValueError(f"expo must be positive, got {self.expo}.")
        # Teleoperation kwargs shared with the keyboard expert.  A task config
        # carries one `expert_kwargs` block for whichever human expert is
        # selected, so swapping EXPERT=spacemouse onto a keyboard-shaped config
        # must not trip the guard below.
        for keyboard_only in ("sticky_steps", "display"):
            kwargs.pop(keyboard_only, None)
        if kwargs:
            raise TypeError(f"Unexpected expert_kwargs for spacemouse: {sorted(kwargs)}")

        try:
            from infra.hardware.spacemouse.spacemouse_expert import SpaceMouseExpert
        except Exception as exc:  # pragma: no cover - depends on host libraries
            raise ImportError(
                "The spacemouse expert needs the HID libraries and a connected "
                "device. Install libhidapi-hidraw0 and make sure /dev/hidraw* "
                "exists, or switch to a scripted expert with "
                "hil.expert=scripted_insert_sim."
            ) from exc

        try:
            self._device = SpaceMouseExpert()
        except Exception as exc:  # pragma: no cover - depends on hardware
            raise RuntimeError(
                "Could not open a SpaceMouse. Check that the device is plugged "
                "in and that this process can read /dev/hidraw*, or switch to a "
                "scripted expert with hil.expert=scripted_insert_sim."
            ) from exc

    def _shape(self, value: float) -> float:
        """Deadzone, then expo, then renormalise to the full output range.

        The rescaling matters as much as the deadzone: clamping small readings
        to zero without it leaves a step discontinuity at the edge, so the
        first motion past the threshold jumps straight to ``deadzone`` and the
        puck feels like it snaps rather than eases in.
        """
        magnitude = abs(float(value))
        if magnitude <= self.deadzone:
            return 0.0
        span = 1.0 - self.deadzone
        normalised = min((magnitude - self.deadzone) / span, 1.0)
        if self.expo != 1.0:
            normalised = normalised ** self.expo
        return math.copysign(normalised, value)

    @property
    def mode_toggle_requested(self) -> bool:
        """True once after the operator clicked the right button; consumed on read."""
        requested, self._mode_toggle_requested = self._mode_toggle_requested, False
        return requested

    def sync_gripper(self, closed: bool) -> None:
        """Adopt the latch the environment is actually holding.

        Called by the wrapper, which sees every executed action and so knows the
        truth even across an ``env.reset`` (both gripper environments reset to
        open).  Without it the operator's belief and the robot's state drift
        apart and a click appears to do nothing.
        """
        self._gripper_closed = bool(closed)

    def _edges(self) -> Tuple[bool, bool]:
        """Rising edges since the last control step, from the reader's counters.

        Counted in the free-running reader process rather than here: a control
        step is 50 ms at 20 Hz, and a crisp click can start and finish between
        two samples of the button level.
        """
        counts = self._device.get_press_counts()
        previous, self._press_counts = self._press_counts, list(counts)
        edges = [
            counts[i] > previous[i] if i < len(previous) else counts[i] > 0
            for i in range(len(counts))
        ]
        left_edge = edges[0] if len(edges) > 0 else False
        right_edge = edges[1] if len(edges) > 1 else False
        return bool(left_edge), bool(right_edge)

    def get_action(self) -> Tuple[np.ndarray, List[int]]:
        raw, buttons = self._device.get_action()
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)

        left = int(bool(buttons[0])) if len(buttons) > 0 else 0
        right = int(bool(buttons[1])) if len(buttons) > 1 else 0
        left_click, right_click = self._edges()

        action = np.zeros(self.action_dim, dtype=np.float32)
        n = min(self.action_dim, raw.shape[0], 6)
        # Per axis, not on the 6-vector's norm: a hand pushing one axis always
        # leans slightly on the others (this device reads ~0.2 of crosstalk on
        # roll/pitch during a pure yaw), and only a per-axis threshold removes
        # that without also killing genuine diagonal motion.
        for axis in range(n):
            scale = self.pos_scale if axis < 3 else self.rot_scale
            action[axis] = self._shape(raw[axis]) * scale

        if self.invert_xy:
            action[0] *= -1.0
            action[1] *= -1.0

        if right_click:
            self._mode_toggle_requested = True

        if self.action_dim > 6:
            if left_click:
                self._gripper_closed = not self._gripper_closed
            action[-1] = (
                -self.gripper_scale if self._gripper_closed else self.gripper_scale
            )

        np.clip(action, -1.0, 1.0, out=action)
        return action, [left, right]

    def reset(self) -> None:
        # Both gripper environments reset their latch to open, so the operator's
        # believed state has to follow or the first click of the new episode is
        # a no-op.  The mode is deliberately *not* reset here: it is the
        # operator's standing choice, and silently taking control back at an
        # episode boundary is exactly what an operator cannot see coming.
        self._gripper_closed = False
        self._mode_toggle_requested = False

    def close(self) -> None:
        self._device.close()
