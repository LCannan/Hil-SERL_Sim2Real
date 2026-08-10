"""Draws the teleoperation HUD from inside the wrapper stack.

Sits directly outside ``ExpertIntervention`` and inside ``RelativeFrame``,
which is the only place both of the things it shows are still readable:

- ``info["intervene_action"]`` is written by the expert in the **base frame**;
  ``RelativeFrame`` rotates it into the end-effector frame on the way out, and
  a number in the TCP frame is not what the operator sees on screen.
- ``obs["images"]`` is still a nested dict here.  ``SERLObsWrapper`` lifts the
  images to the top level and ``ChunkingWrapper`` adds a horizon axis.

Rendering is driven from ``step``/``reset`` rather than a display thread
because the cv2 window doubles as the keyboard backend's key source, and
``waitKey`` only reports to the thread owning the window.
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation


class TeleopHUD(gym.Wrapper):
    """Render camera views plus operator telemetry.  Observations pass through.

    A no-op when ``display`` is unavailable, so the same stack is built
    headless without branching at the call site.
    """

    def __init__(self, env: gym.Env, display: Any, target_pose=None):
        super().__init__(env)
        self.display = display
        self._target_pose = (
            None if target_pose is None
            else np.asarray(target_pose, dtype=np.float64).reshape(-1)
        )
        self._step = 0
        self._episode = 0
        self._return = 0.0

    @property
    def _active(self) -> bool:
        return self.display is not None and getattr(
            self.display, "available", False
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step = 0
        self._episode += 1
        self._return = 0.0
        self._render(obs, info, action=None, intervening=False, succeed=False)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        self._return += float(reward)

        executed = info.get("intervene_action")
        intervening = executed is not None
        self._render(
            obs,
            info,
            action=executed if intervening else action,
            intervening=intervening,
            succeed=bool(info.get("succeed", False)),
        )
        return obs, reward, terminated, truncated, info

    def _render(self, obs, info, action, intervening: bool, succeed: bool):
        if not self._active:
            return
        frames = obs.get("images") if isinstance(obs, dict) else None
        if not frames:
            return
        self.display.render(
            frames,
            {
                "intervening": intervening,
                "action": action,
                "pose_error": self._pose_error(obs),
                "step": self._step,
                "episode": self._episode,
                "episode_return": self._return,
                "succeed": succeed,
            },
        )

    def _pose_error(self, obs) -> Optional[np.ndarray]:
        """Distance to the goal pose, the operator's main aiming cue.

        Mirrors the environment's own reward test, which fires only when every
        one of these six terms is under its threshold.
        """
        if self._target_pose is None or self._target_pose.shape[0] < 7:
            return None
        state = obs.get("state") if isinstance(obs, dict) else None
        if not isinstance(state, dict) or "tcp_pose" not in state:
            return None
        tcp = np.asarray(state["tcp_pose"], dtype=np.float64).reshape(-1)
        if tcp.shape[0] < 7:
            return None

        position = tcp[:3] - self._target_pose[:3]
        rotation = (
            Rotation.from_quat(tcp[3:7])
            * Rotation.from_quat(self._target_pose[3:7]).inv()
        ).as_rotvec()
        return np.concatenate([position, rotation])

    def close(self):
        try:
            if self.display is not None:
                self.display.close()
        finally:
            return self.env.close()
