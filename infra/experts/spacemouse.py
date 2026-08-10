"""Adapter exposing the existing SpaceMouse driver through the Expert protocol.

Thin by design: :class:`~infra.hardware.spacemouse.spacemouse_expert.SpaceMouseExpert`
already has the right ``get_action``/``close`` signatures.  This wrapper adds
only an actionable import error and a fixed 2-element button tuple.
"""

from typing import Any, List, Tuple

import numpy as np

from .base import ExpertBase


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
        **kwargs: Any,
    ):
        # `seed` is accepted and ignored, matching the scripted experts: the
        # factory seeds every expert uniformly, and without this the guard
        # below rejects it and `hil.expert=spacemouse` cannot be built at all.
        del env, seed  # a human needs neither privileged state nor a seed
        self.action_dim = int(action_dim)
        self.gripper_scale = float(kwargs.pop("gripper_scale", 1.0))
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

    def get_action(self) -> Tuple[np.ndarray, List[int]]:
        raw, buttons = self._device.get_action()
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)

        # The driver reports 6 Cartesian axes.  Environments with a gripper
        # dimension get it from the buttons, matching SpacemouseIntervention's
        # convention (left closes, right opens).
        left = int(bool(buttons[0])) if len(buttons) > 0 else 0
        right = int(bool(buttons[1])) if len(buttons) > 1 else 0

        action = np.zeros(self.action_dim, dtype=np.float32)
        n = min(self.action_dim, raw.shape[0])
        action[:n] = raw[:n]
        if self.action_dim > 6:
            gripper = 0.0
            if left:
                gripper = -self.gripper_scale
            elif right:
                gripper = self.gripper_scale
            action[-1] = gripper

        return action, [left, right]

    def close(self) -> None:
        self._device.close()
