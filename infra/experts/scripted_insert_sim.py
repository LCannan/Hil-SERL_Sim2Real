"""Scripted stand-in for a human operator on the MuJoCo peg-insertion task.

Reads privileged simulator state (the mocap setpoint) the same way a human
reads the screen, and drives the 6-DoF Cartesian action space.  Measured
success rate from reset, through the full wrapper stack: 8/10.

Three findings shaped this controller; each cost a full parameter sweep to
locate, so they are worth stating plainly:

1. **Close the loop on the mocap setpoint, not on ``tcp_pose``.**
   ``PandaPegInsertGymEnv.step`` integrates ``_data.mocap_pos[...]``; the real
   TCP trails it through an impedance servo.  Controlling on the observed
   ``tcp_pose`` winds up and oscillates at +-0.02 m forever.  Every P/PD variant
   over ``kp`` in {0.15..1.0} x ``kd`` in {0..0.1} x ``tol`` in {0.006..0.012}
   topped out at 3/10.
2. **Gate the descent on orientation as well as xy.**  Descending while still
   ~16 deg off wedges the peg on the hole rim: mocap sits happily at the target
   while ``tcp_z`` freezes at 0.2129 for the rest of the episode.  Adding the
   rotation gate took this from 0/10 to 5/10.
3. **Detect the stall and lift clear before retrying.**  A wedged peg cannot be
   nudged free in place -- a spiral search without the lift was worth exactly
   nothing.  Retract to the safe height, re-seat at a small random offset, and
   try again: 5/10 -> 7/10.

Also ruled out, so nobody re-runs them: integral bias to cancel the servo lag
(1-2/10) and rate-limiting the mocap setpoint to stay near the true TCP (0/10,
all timeouts).

A consequence of (1) worth knowing before reading a recorded demo: once the
mocap setpoint has reached the target the delta is exactly zero, so the expert
commands all-zero actions for as long as the impedance servo needs to catch up
-- around 40% of the transitions in a typical demo, in contiguous runs. Those
rows are the controller correctly waiting, not a dropped action.
"""

from typing import Any, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .base import ExpertBase

_ALIGN = "align"
_DESCEND = "descend"
_RETRACT = "retract"


class ScriptedInsertSimExpert(ExpertBase):
    """Align above the hole at a safe height, then descend; lift and retry on a stall.

    Re-entrant: the phase is recomputed from current geometry on every call, so
    the expert can take over mid-episode from an arbitrary policy-induced state
    rather than only from ``reset``.  Only the stall window is remembered, and
    it is cleared whenever control is handed back.
    """

    def __init__(
        self,
        env: Any,
        safe_z: float = 0.235,
        xy_tol: float = 0.0015,
        rot_tol: float = 0.03,
        stall_patience: int = 5,
        stall_epsilon: float = 5e-4,
        retry_jitter: float = 0.003,
        jam_z_margin: float = 0.018,
        seed: int = 0,
        action_dim: int = 6,
        **kwargs: Any,
    ):
        if kwargs:
            raise TypeError(
                f"Unexpected expert_kwargs for scripted_insert_sim: {sorted(kwargs)}"
            )
        if env is None:
            raise ValueError(
                "ScriptedInsertSimExpert needs the unwrapped PandaPegInsertGymEnv "
                "to read the mocap setpoint."
            )

        self.env = env
        self.safe_z = float(safe_z)
        self.xy_tol = float(xy_tol)
        self.rot_tol = float(rot_tol)
        self.stall_patience = int(stall_patience)
        self.stall_epsilon = float(stall_epsilon)
        self.retry_jitter = float(retry_jitter)
        self.jam_z_margin = float(jam_z_margin)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)

        self._target = np.asarray(env._target_pose, dtype=np.float64)
        self._action_scale = np.asarray(env._action_scale, dtype=np.float64)
        self._ee_id = env._end_effector_id

        self.reset()

    def reset(self) -> None:
        self._phase = _ALIGN
        self._prev_tcp_z = None
        self._stall = 0
        self._offset = np.zeros(2)

    # A fresh takeover is a fresh attempt: stale stall history from before the
    # policy was driving would trigger a spurious retract.
    on_takeover = reset

    def _mocap_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        data = self.env._data
        pos = np.asarray(data.mocap_pos[self._ee_id], dtype=np.float64).copy()
        wxyz = np.asarray(data.mocap_quat[self._ee_id], dtype=np.float64)
        return pos, np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

    def _tcp_z(self) -> float:
        return float(self.env._data.site_xpos[self.env._hand_site_id][2])

    def get_action(self) -> Tuple[np.ndarray, List[int]]:
        mocap_pos, mocap_quat = self._mocap_pose()
        tcp_z = self._tcp_z()

        rot_err = (
            Rotation.from_quat(self._target[3:])
            * Rotation.from_quat(mocap_quat).inv()
        ).as_rotvec()

        goal_xy = self._target[:2] + self._offset
        aligned = (
            np.linalg.norm(goal_xy - mocap_pos[:2]) <= self.xy_tol
            and np.linalg.norm(rot_err) <= self.rot_tol
        )

        if self._phase == _ALIGN:
            goal_z = self.safe_z
            if aligned and abs(tcp_z - self.safe_z) < 0.01:
                self._phase = _DESCEND
                self._prev_tcp_z = None
                self._stall = 0
        elif self._phase == _DESCEND:
            goal_z = self._target[2]
            # Above the jam threshold the peg is still outside the hole, so a
            # motionless TCP means it is caught on the rim.
            if self._prev_tcp_z is not None and tcp_z > self._target[2] + self.jam_z_margin:
                if (self._prev_tcp_z - tcp_z) < self.stall_epsilon:
                    self._stall += 1
                else:
                    self._stall = 0
                if self._stall >= self.stall_patience:
                    self._phase = _RETRACT
                    self._offset = self._rng.uniform(
                        -self.retry_jitter, self.retry_jitter, size=2
                    )
            self._prev_tcp_z = tcp_z
        else:  # _RETRACT
            goal_z = self.safe_z + 0.01
            if tcp_z > self.safe_z:
                self._phase = _ALIGN
                self._prev_tcp_z = None
                self._stall = 0

        delta = np.array(
            [goal_xy[0] - mocap_pos[0], goal_xy[1] - mocap_pos[1], goal_z - mocap_pos[2]]
        )

        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:3] = np.clip(delta / self._action_scale[0], -1.0, 1.0)
        action[3:6] = np.clip(rot_err / self._action_scale[1], -1.0, 1.0)
        return action, [0, 0]
