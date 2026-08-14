"""A human driving the robot from the keyboard.

Stands in for a SpaceMouse on machines that do not have one.  Output goes
through the same ``Expert`` contract, so ``hil.trigger=manual`` gates it
exactly as it gates the device: keys held produce a non-zero action and the
wrapper hands over; keys released produce zeros, the norm drops under
``manual_deadband``, and the policy gets control back with no extra state.

Two backends, chosen at construction:

``pynput``
    Real press/release events, so an action lasts exactly as long as the key
    is held.  Needs the package plus a reachable X server.

``cv2``
    Falls back to the teleop window's ``waitKey``.  That reports *repeats*,
    not holds, so a keypress is latched for ``sticky_steps`` control steps and
    decays -- closer to jogging than to holding, which suits an insertion task
    made of small corrections.  Requires a :class:`TeleopDisplay`.

Both are skipped when neither input is reachable (a headless learner, say);
the expert then returns zeros forever and never intervenes.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ExpertBase

# Axis, sign, and which scale applies.  Base-frame throughout: the expert is
# mounted innermost, so what a key does here is what the operator sees on
# screen -- RelativeFrame only rotates the *recorded* action, afterwards.
_BINDINGS = {
    "w": (0, +1.0, "pos"),
    "s": (0, -1.0, "pos"),
    "a": (1, +1.0, "pos"),
    "d": (1, -1.0, "pos"),
    "q": (2, +1.0, "pos"),
    "e": (2, -1.0, "pos"),
    "i": (3, +1.0, "rot"),
    "k": (3, -1.0, "rot"),
    "j": (4, +1.0, "rot"),
    "l": (4, -1.0, "rot"),
    "u": (5, +1.0, "rot"),
    "o": (5, -1.0, "rot"),
}

KEY_LEGEND = "WASD/QE move  IJKL/UO rotate  shift=fine  space=grip  r=reset"

_FINE_SCALE = 0.25


class _NullBackend:
    """No input reachable -- the expert exists but never intervenes."""

    name = "null"

    def held(self) -> Dict[str, float]:
        return {}

    def close(self) -> None:
        return None


class _PynputBackend:
    """True press/release tracking via a background listener."""

    name = "pynput"

    def __init__(self):
        # Guard the import itself, not just its use: with libX11 present but
        # no DISPLAY, pynput raises while being imported.
        from pynput import keyboard

        self._keyboard = keyboard
        self._held = set()
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    def _name_of(self, key) -> Optional[str]:
        char = getattr(key, "char", None)
        if char:
            return char.lower()
        if key in (self._keyboard.Key.shift, self._keyboard.Key.shift_r):
            return "shift"
        if key == self._keyboard.Key.space:
            return "space"
        return None

    def _on_press(self, key) -> None:
        name = self._name_of(key)
        if name:
            self._held.add(name)

    def _on_release(self, key) -> None:
        self._held.discard(self._name_of(key))

    def held(self) -> Dict[str, float]:
        return {name: 1.0 for name in tuple(self._held)}

    def close(self) -> None:
        self._listener.stop()


class _Cv2Backend:
    """Latch each keypress for a few steps, since waitKey cannot see holds."""

    name = "cv2"

    def __init__(self, display: Any, sticky_steps: int):
        self._display = display
        self._sticky = max(1, int(sticky_steps))
        self._remaining: Dict[str, int] = {}

    def held(self) -> Dict[str, float]:
        code = self._display.poll_key()
        if code != -1:
            char = chr(code).lower() if 32 <= code < 127 else ""
            if char == " ":
                char = "space"
            if char:
                self._remaining[char] = self._sticky

        active = {}
        for name, left in tuple(self._remaining.items()):
            # Fade the action out over the latch window rather than cutting it
            # dead, so a tap reads as a nudge instead of a step change.
            active[name] = left / self._sticky
            if left <= 1:
                self._remaining.pop(name)
            else:
                self._remaining[name] = left - 1
        return active

    def close(self) -> None:
        self._remaining.clear()


class KeyboardExpert(ExpertBase):
    """Keyboard teleoperation as an :class:`~infra.experts.base.Expert`.

    Emits an action of the environment's full width.  Dimensions past the six
    Cartesian axes stay zero except a gripper in the last slot, which only
    exists when the environment has one (``action_dim > 6``); ``insert_sim``
    does not -- its peg is welded to the finger.
    """

    def __init__(
        self,
        env: Any = None,
        action_dim: int = 6,
        seed: int = 0,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        sticky_steps: int = 6,
        display: Any = None,
        invert_xy: bool = False,
        **kwargs: Any,
    ):
        del env, seed  # a human needs neither privileged state nor a seed
        # Mirror of the pops in SpaceMouseAdapter: one `expert_kwargs` block
        # serves whichever human expert is selected, so EXPERT=keyboard on a
        # spacemouse-tuned config must not trip the guard below.  A keyboard has
        # no analogue deflection, hence nothing for these to shape.
        for spacemouse_only in ("deadzone", "expo", "gripper_scale"):
            kwargs.pop(spacemouse_only, None)
        if kwargs:
            raise TypeError(
                f"Unexpected expert_kwargs for keyboard: {sorted(kwargs)}"
            )
        self.action_dim = int(action_dim)
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        # See SpaceMouseAdapter: a scene camera facing the robot mirrors the
        # horizontal plane, so the key for "right" moves the arm left on screen.
        self.invert_xy = bool(invert_xy)
        self._gripper = 0.0
        self._reset_requested = False
        self._backend = self._make_backend(display, sticky_steps)

    @staticmethod
    def _make_backend(display: Any, sticky_steps: int):
        try:
            return _PynputBackend()
        except Exception:
            # Not installed, or no reachable X server.  Either way the cv2
            # window is the remaining option.
            pass
        if display is not None and getattr(display, "available", False):
            return _Cv2Backend(display, sticky_steps)
        return _NullBackend()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def reset_requested(self) -> bool:
        """True once after the operator pressed ``r``; consumed on read."""
        requested, self._reset_requested = self._reset_requested, False
        return requested

    def get_action(self) -> Tuple[np.ndarray, List[int]]:
        held = self._backend.held()
        action = np.zeros(self.action_dim, dtype=np.float32)

        if "r" in held:
            self._reset_requested = True
        fine = _FINE_SCALE if "shift" in held else 1.0

        for name, weight in held.items():
            binding = _BINDINGS.get(name)
            if binding is None:
                continue
            axis, sign, kind = binding
            if axis >= self.action_dim:
                continue
            scale = self.pos_scale if kind == "pos" else self.rot_scale
            action[axis] += sign * scale * fine * float(weight)

        if self.invert_xy:
            action[0] *= -1.0
            action[1] *= -1.0

        # Space toggles the gripper, and only where one exists.
        left = right = 0
        if self.action_dim > 6:
            if "space" in held:
                self._gripper = -1.0 if self._gripper >= 0.0 else 1.0
            action[-1] = self._gripper
            left = int(self._gripper < 0.0)
            right = int(self._gripper > 0.0)

        np.clip(action, -1.0, 1.0, out=action)
        return action, [left, right]

    def reset(self) -> None:
        self._gripper = 0.0
        self._reset_requested = False

    def close(self) -> None:
        self._backend.close()
