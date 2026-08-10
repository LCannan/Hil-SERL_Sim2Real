"""SERL-facing gym env for the ManiSkill PegInsertionSide task.

The action space is 8-dim ``pd_joint_delta_pos`` and the proprioceptive state is
the Panda's 9-dim ``qpos`` -- both matching the
``RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ`` dataset, so demonstrations
converted by ``train/convert_lerobot_demo.py`` are directly usable here.

Nothing at module scope imports ``mani_skill`` or ``sapien``: the learner builds
this env with ``fake_env=True`` purely to read the observation/action spaces, and
the demo converter imports :func:`resize_rgb` from here.  Both must work on a
host that never installed the optional ManiSkill extra.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import cv2
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig

from infra.sim.envs.maniskill_peg_variants import (
    MAIN_CAMERA_KEY,
    PANDA_WIDE_WRISTCAM_UID,
    WRIST_CAMERA_KEY,
    register_maniskill_peg_variants,
)
from infra.utils.config_util import as_array

# Panda arm (7) + the two finger joints.
_QPOS_DIM = 9
# ``pd_joint_delta_pos``: 7 arm deltas + 1 gripper command, already normalised to
# [-1, 1] by ManiSkill's PDJointPosControllerConfig.
_ACTION_DIM = 8
# ``make_sac_pixel_agent`` does not forward an image size, so the encoder is
# always built for SACAgent.create_pixels' 128x128 default.
_ENCODER_IMAGE_SIZE = 128
_DISPLAY_WINDOW = "maniskill_peg_insert"


def resize_rgb(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Downsample an RGB frame to ``size``, given as ``(height, width)``.

    Shared by this environment *and* ``train/convert_lerobot_demo.py`` so that
    demo images and online images pass through identical processing by
    construction rather than by convention.  Do not move this helper into a
    module that imports mani_skill -- the converter must run without the extra.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"resize_rgb expects an (H, W, 3) RGB image, got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        # Catches the float-image-silently-cast-to-uint8-zeros failure at the
        # cheapest possible place.
        raise TypeError(
            f"resize_rgb expects a uint8 RGB image, got dtype {array.dtype}"
        )

    height, width = int(size[0]), int(size[1])
    if array.shape[:2] == (height, width):
        return array
    # cv2.resize takes (width, height).  INTER_AREA is the correct kernel for
    # decimation; INTER_LINEAR aliases away the thin-peg detail the task turns
    # on.  No BGR swap -- SAPIEN sensors and PIL both hand back RGB already.
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)


def _int_pair(value: Any, name: str) -> Tuple[int, int]:
    array = as_array(value, name, ((2,),))
    if not np.array_equal(array, np.round(array)):
        raise ValueError(f"{name} must contain integers, got {value}")
    return int(array[0]), int(array[1])


class ManiSkillPegInsertGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: DictConfig,
        seed: int = 0,
        fake_env: bool = False,
    ):
        if config is None:
            raise ValueError("ManiSkillPegInsertGymEnv requires an environment config")

        self.config = config
        self.fake_env = bool(fake_env)
        self._seed = int(seed)
        self._seeded = False

        self._env_id = str(config.env_id)
        self._control_mode = str(config.control_mode)
        self._sim_backend = str(config.sim_backend)
        self._sim_freq = int(config.sim_freq)
        self._control_freq = int(config.control_freq)
        self._render_size = _int_pair(config.render_size, "render_size")
        self._image_size = _int_pair(config.image_size, "image_size")
        self._max_episode_length = int(config.max_episode_length)
        self._sparse_reward_on_success = float(config.sparse_reward_on_success)
        self.display_image = bool(config.display_image)

        if self._max_episode_length <= 0:
            raise ValueError("max_episode_length must be positive")
        if self._image_size != (_ENCODER_IMAGE_SIZE, _ENCODER_IMAGE_SIZE):
            # Any other value constructs fine and only dies on the first forward
            # pass with an opaque tensor-size mismatch, so reject it up front.
            raise ValueError(
                f"image_size must be [{_ENCODER_IMAGE_SIZE}, {_ENCODER_IMAGE_SIZE}]: "
                "the SAC pixel encoder is built for that size and does not take "
                f"an override.  Got {list(self._image_size)}."
            )

        self.render_mode = "rgb_array"
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": self._control_freq,
        }
        self.cur_episode_length = 0

        height, width = self._image_size
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "qpos": gym.spaces.Box(
                            -np.inf, np.inf, shape=(_QPOS_DIM,), dtype=np.float32
                        ),
                    }
                ),
                # Raw SAPIEN sensor names are used verbatim as SERL image keys so
                # the env and the demo converter cannot drift apart via a rename.
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0, 255, shape=(height, width, 3), dtype=np.uint8
                        )
                        for key in (MAIN_CAMERA_KEY, WRIST_CAMERA_KEY)
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            -np.ones((_ACTION_DIM,), dtype=np.float32),
            np.ones((_ACTION_DIM,), dtype=np.float32),
            dtype=np.float32,
        )

        self._env = None
        self._display_open = False
        if self.fake_env:
            # The learner only ever needs the spaces above; returning here keeps
            # mani_skill and sapien out of its process entirely.
            return

        self._env = self._make_env()

    def _make_env(self):
        register_maniskill_peg_variants()

        import gymnasium

        from mani_skill.utils.wrappers import CPUGymWrapper

        render_height, render_width = self._render_size
        env = gymnasium.make(
            self._env_id,
            num_envs=1,
            obs_mode="rgb",
            control_mode=self._control_mode,
            reward_mode="sparse",
            render_mode="rgb_array",
            sim_backend=self._sim_backend,
            sim_config={
                "sim_freq": self._sim_freq,
                "control_freq": self._control_freq,
            },
            sensor_configs={
                "shader_pack": "default",
                "height": render_height,
                "width": render_width,
            },
            max_episode_steps=self._max_episode_length,
            robot_uids=PANDA_WIDE_WRISTCAM_UID,
        )
        # CPUGymWrapper asserts num_envs == 1, adopts the single_* spaces and
        # does the unbatch + torch->numpy conversion.  record_metrics stays off:
        # RecordEpisodeStatistics in train_serl.py already supplies info["episode"].
        return CPUGymWrapper(env, record_metrics=False)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.cur_episode_length = 0
        if self.fake_env:
            return self._zero_observation(), {}

        if seed is None and not self._seeded:
            seed = self._seed
        self._seeded = True

        obs, _ = self._env.reset(seed=seed, **kwargs)
        observation = self._extract_observation(obs)
        self._maybe_display(observation["images"])
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self.cur_episode_length += 1

        if self.fake_env:
            return self._zero_observation(), 0.0, False, False, {"succeed": False}

        obs, _, _, _, info = self._env.step(action)

        succeed = bool(np.asarray(info["success"]).reshape(-1)[0])
        # Recomputed rather than taken from ManiSkill's reward so that it is
        # exactly what the demo converter synthesizes -- one config knob drives
        # both sides.
        reward = self._sparse_reward_on_success if succeed else 0.0
        terminated = succeed
        # Own the time limit here instead of trusting ManiSkill's MSTimeLimit.
        truncated = self.cur_episode_length >= self._max_episode_length and not terminated

        observation = self._extract_observation(obs)
        self._maybe_display(observation["images"])
        # ManiSkill's info carries large per-step arrays that record_demo.py
        # would deepcopy into every stored transition.
        return observation, reward, terminated, truncated, {"succeed": succeed}

    def render_camera(
        self,
        camera_name: str,
        width: int = 256,
        height: int = 256,
        mode: Literal["rgb_array"] = "rgb_array",
    ) -> np.ndarray:
        """Render an extra camera without adding it to policy observations.

        Signature matches ``MujocoGymEnv.render_camera`` so ``train_serl.py``'s
        evaluation-video path works unchanged.  Note this scene's human render
        camera is named ``render_camera``, not the ``front`` default -- pass
        ``--eval_video_main_camera=render_camera``.
        """
        if mode != "rgb_array":
            raise ValueError(
                f"ManiSkillPegInsertGymEnv only renders rgb_array, got {mode!r}"
            )
        if self.fake_env or self._env is None:
            return np.zeros((height, width, 3), dtype=np.uint8)

        from mani_skill.utils import common

        base = self._env.unwrapped
        cameras = getattr(base, "_human_render_cameras", None) or {}
        if camera_name not in cameras:
            raise ValueError(
                f"Unknown render camera {camera_name!r}. Available: "
                f"{sorted(cameras)}.  Pass --eval_video_main_camera=render_camera."
            )

        frame = common.to_numpy(base.render_rgb_array(camera_name=camera_name))
        frame = np.asarray(common.unbatch(frame), dtype=np.uint8)
        return resize_rgb(frame[..., :3], (height, width))

    def close(self) -> None:
        env, self._env = self._env, None
        if env is not None:
            env.close()
        if self._display_open:
            cv2.destroyWindow(_DISPLAY_WINDOW)
            self._display_open = False

    def _extract_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # obs_mode="rgb" leaves `agent` un-flattened, so qpos stays addressable.
        qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)
        if qpos.shape[0] < _QPOS_DIM:
            raise ValueError(
                f"Expected at least {_QPOS_DIM} qpos values, got {qpos.shape[0]}"
            )

        sensor_data = obs["sensor_data"]
        images = {}
        for key in (MAIN_CAMERA_KEY, WRIST_CAMERA_KEY):
            if key not in sensor_data:
                raise KeyError(
                    f"Camera {key!r} missing from the observation. Available: "
                    f"{sorted(sensor_data)}"
                )
            images[key] = resize_rgb(
                np.asarray(sensor_data[key]["rgb"]), self._image_size
            )

        return {"state": {"qpos": qpos[:_QPOS_DIM]}, "images": images}

    def _zero_observation(self) -> Dict[str, Any]:
        # Derived from the spaces so it cannot drift away from them.
        return {
            "state": {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space["state"].spaces.items()
            },
            "images": {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space["images"].spaces.items()
            },
        }

    def _maybe_display(self, images: Dict[str, np.ndarray]) -> None:
        if not self.display_image:
            return
        frame = np.concatenate(
            [images[MAIN_CAMERA_KEY], images[WRIST_CAMERA_KEY]], axis=1
        )
        cv2.imshow(_DISPLAY_WINDOW, frame[..., ::-1])  # cv2 windows want BGR
        cv2.waitKey(1)
        self._display_open = True
