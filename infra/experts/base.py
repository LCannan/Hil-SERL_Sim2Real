"""Expert interface for human-in-the-loop training.

An "expert" is whatever supplies the corrective action during a HIL-SERL run:
a human on a SpaceMouse, or a scripted controller standing in for one.  Both
satisfy :class:`Expert`, so :class:`~infra.wrappers.intervention.ExpertIntervention`
does not care which is plugged in.
"""

from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class Expert(Protocol):
    """Source of corrective actions.

    The signature deliberately matches the pre-existing
    :class:`~infra.hardware.spacemouse.spacemouse_expert.SpaceMouseExpert` so the
    hardware adapter is nearly free.

    Scripted experts are handed the *unwrapped* environment and may read
    privileged simulator state (mocap targets, object poses).  That is
    legitimate: the expert stands in for a human, who also sees more than the
    policy's cameras do.  Expert output never enters the agent's observation.
    """

    def reset(self) -> None:
        """Clear per-episode state.  Called from the wrapper's ``reset``."""
        ...

    def get_action(self) -> Tuple[np.ndarray, List[int]]:
        """Return ``(action, buttons)`` for the current state.

        ``action`` has the environment's full action dimensionality -- unlike
        ``SpacemouseIntervention``, the new wrapper never appends a gripper
        dimension.  ``buttons`` is exactly two ints ``[left, right]``.
        """
        ...

    def close(self) -> None:
        """Release any resources (device handles, planner processes)."""
        ...


class ExpertBase:
    """Optional base with no-op lifecycle methods.

    Subclasses only have to implement :meth:`get_action`.
    """

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_action(self) -> Tuple[np.ndarray, List[int]]:
        raise NotImplementedError
