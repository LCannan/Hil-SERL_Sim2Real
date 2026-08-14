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

    def __init__(self, env: gym.Env, display: Any, target_pose=None,
                 render_size: int = 0, cameras=None, show_policy_view: bool = False):
        super().__init__(env)
        self.display = display
        self._target_pose = (
            None if target_pose is None
            else np.asarray(target_pose, dtype=np.float64).reshape(-1)
        )
        # Re-render the operator's view at this size instead of upscaling the
        # policy's 128x128 observation.  The blur on screen is a resolution
        # limit, not an interpolation artefact -- no filter recovers a wrist
        # camera in which the gripper itself is four pixels wide.  Measured on
        # this task, two extra renders at 640 cost ~7 ms/step against a 50 ms
        # budget at 20 Hz, so this buys the detail almost for free.
        self._render_size = int(render_size)
        self._render_env = self._find_render_env(env) if self._render_size else None
        # The operator's cameras are named explicitly rather than taken from the
        # observation, so that adding a camera to `training.image_keys` changes
        # what the *policy* sees without rearranging the view the operator has
        # learned to fly by.  None keeps the old behaviour of following the
        # observation.
        self._cameras = list(cameras) if cameras else None
        # Show the policy's own frames underneath, at their true size.  The two
        # rows are not the same picture: the top is re-rendered for the human,
        # the bottom is the 128x128 the network actually receives.
        self._show_policy_view = bool(show_policy_view)
        self._step = 0
        self._episode = 0
        self._return = 0.0

    @staticmethod
    def _find_render_env(env):
        """Walk in to whichever wrapper owns `render_camera`."""
        node = env
        while node is not None:
            if hasattr(node, "render_camera"):
                return node
            node = getattr(node, "env", None)
        return None

    def _hud_frames(self, obs):
        """The operator's view: re-rendered when possible, else the policy's."""
        frames = obs.get("images") if isinstance(obs, dict) else None
        if not frames or self._render_env is None:
            return frames
        names = self._cameras or list(frames.keys())
        size = self._render_size
        try:
            return {
                name: self._render_env.render_camera(
                    name, width=size, height=size
                )
                for name in names
            }
        except Exception as exc:
            # A camera name the scene does not carry, or a renderer that cannot
            # serve a second pass.  The HUD is a convenience -- degrade to the
            # policy's own frames rather than take the episode down with it.
            print(f"[hil] HUD re-render disabled: {exc}")
            self._render_env = None
            return frames

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
        frames = self._hud_frames(obs)
        if not frames:
            return
        policy_frames = None
        if self._show_policy_view:
            policy_frames = obs.get("images") if isinstance(obs, dict) else None
            # Only worth a second row when it differs from the top one; with no
            # re-render the two would be the same picture twice.
            if policy_frames is frames:
                policy_frames = None
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
                "hil_mode": info.get("hil_mode"),
                "gripper_closed": info.get("gripper_closed"),
            },
            policy_frames=policy_frames,
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
