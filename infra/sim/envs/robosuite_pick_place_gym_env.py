"""SERL-facing gym env for robosuite's ``PickPlaceMilk``.

Pick a milk carton out of the source bin and place it in its quadrant of the
target bin.  Structurally this follows :mod:`maniskill_peg_gym_env` rather than
the native-MuJoCo ``panda_*`` envs: robosuite is a third-party simulator that
owns its own scene, so the job here is adaptation, not physics.  Nothing at
module scope imports ``robosuite`` -- the learner builds this env with
``fake_env=True`` purely to read the observation/action spaces.

Five robosuite behaviours differ from what the rest of this repo assumes, and
all five are silent failures rather than crashes:

- **The gripper convention is inverted and integrating.**  robosuite's
  ``PandaGripper.format_action`` reads only ``sign(action)`` and accumulates at
  ``speed=0.2``, so ``+1`` *closes* over ~5 steps and ``-1`` opens.  This repo's
  convention (``pick_cube_sim``) is the opposite sign and *latched*: ``< -0.5``
  closes, ``> 0.5`` opens, anything between holds.  We keep the repo convention
  and translate, holding the latched command down every step -- which is exactly
  what drives an integrating actuator to its endpoint and keeps it there through
  a long carry, when the operator is pushing translation only and the gripper
  channel reads ~0.
- **Images come back upside-down.**  ``macros.IMAGE_CONVENTION`` defaults to
  ``"opengl"``, so every frame is vertically flipped relative to what a person
  would call upright (verified visually, not assumed).  A policy trained on the
  raw frames learns fine and looks fine -- it is simply learning an inverted
  world, and will not transfer to any other source of images.
- **Success does not terminate the episode.**  robosuite's ``done`` is
  ``timestep >= horizon`` and nothing else, so ``_check_success()`` has to be
  polled.  We construct with ``ignore_done=True`` and own the time limit here,
  for the same reason the ManiSkill env ignores ``MSTimeLimit``: two truncation
  clocks racing each other is worse than one.
- **Success requires releasing *and retreating*.**  ``_check_success`` demands
  the object be in its bin quadrant *and* ``r_reach = 1 - tanh(10*d) < 0.6``,
  i.e. the gripper more than **4.24 cm** away.  A demonstration that carries the
  milk into the bin and holds it there scores exactly zero.
- **Old-gym API.**  ``reset()`` returns a bare obs and ``step()`` a 4-tuple.

Reward is sparse and comes from ``_check_success`` unchanged.  ``reward_shaping``
is deliberately off: under ``single_object_mode=2`` the three unused objects are
teleported to ``(10, 10, 10)`` but still participate in ``staged_rewards``, whose
``r_lift`` term takes ``min`` over all of them -- so it saturates the instant
anything is grasped and carries no gradient through the lift.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import cv2
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig

# The latch thresholds are shared with the discrete grasp critic, which has to
# label recorded actions with exactly the semantics executed here -- a mismatch
# would mislabel training data rather than raise.  `algorithm.utils.gripper` is
# dependency-free (torch + numpy) so importing it does not pull the learner's
# stack into an env module, nor vice versa.
from algorithm.utils.gripper import CLOSE_THRESHOLD as _GRIPPER_CLOSE_THRESHOLD
from algorithm.utils.gripper import OPEN_THRESHOLD as _GRIPPER_OPEN_THRESHOLD
from infra.utils.config_util import as_array

# robosuite's own key names, used verbatim as SERL image keys so the env and any
# future demo converter cannot drift apart via a rename.
_AGENT_CAMERA = "agentview"
_WRIST_CAMERA = "robot0_eye_in_hand"

# `make_sac_pixel_agent` does not forward an image size, so the encoder is always
# built for SACAgent.create_pixels' 128x128 default.
_ENCODER_IMAGE_SIZE = 128
_DISPLAY_WINDOW = "robosuite_pick_place"

# robosuite's gripper polarity, which is the reverse of this repo's.
_ROBOSUITE_CLOSE = 1.0
_ROBOSUITE_OPEN = -1.0

# PandaGripper finger travel, for normalising `gripper_pose` into [0, 1].
_FINGER_QPOS_OPEN = 0.04

_ARM_NAME = "right"

_ASSET_DIR = Path(__file__).parent / "assets" / "robosuite_objects"

# Single source of truth for the proprioceptive layout.  `_zero_observation`
# builds from this rather than from `self.observation_space`, because
# Quat2RotvecWrapper rewrites that space's `tcp_pose` entry in place.
_STATE_DIMS = {
    # 7-dim and xyzw, which is what RelativeFrame and Quat2RotvecWrapper both
    # hard-require.  Built from `robot0_eef_pos` plus `robot0_eef_quat_site` --
    # the site-derived quaternion, so position and orientation describe the same
    # frame (`robot0_eef_quat` is the body's).
    "tcp_pose": 7,
    "tcp_vel": 6,
    "tcp_force": 3,
    "tcp_torque": 3,
    # Normalised finger opening, so the policy can tell a closed-and-holding
    # hand from a closed-and-empty one when combined with the force reading.
    "gripper_pose": 1,
    # Privileged, and deliberately so: this is a sim task whose point is
    # validating the HIL plumbing, not learning object pose from pixels alone.
    "object_pose": 7,
    "object_to_goal": 3,
}


def _int_pair(value: Any, name: str) -> Tuple[int, int]:
    array = as_array(value, name, ((2,),))
    if not np.array_equal(array, np.round(array)):
        raise ValueError(f"{name} must contain integers, got {value}")
    return int(array[0]), int(array[1])


def _look_at_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo camera quaternion (wxyz) placing ``pos`` looking at ``target``.

    A MuJoCo camera looks down its own -z with +y up, so the rotation is the
    frame whose columns are (right, up, -look).
    """
    from scipy.spatial.transform import Rotation

    look = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    look /= np.linalg.norm(look)
    right = np.cross(look, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, look)
    x, y, z, w = Rotation.from_matrix(
        np.stack([right, up, -look], axis=1)
    ).as_quat()
    return np.array([w, x, y, z])


class RobosuitePickPlaceGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: DictConfig,
        seed: int = 0,
        fake_env: bool = False,
    ):
        if config is None:
            raise ValueError("RobosuitePickPlaceGymEnv requires an environment config")

        self.config = config
        self.fake_env = bool(fake_env)
        self._seed = int(seed)
        self._seeded = False

        self._env_name = str(config.env_name)
        self._robot = str(config.robot)
        self._controller = str(config.controller)
        self._control_freq = int(config.control_freq)
        self._image_size = _int_pair(config.image_size, "image_size")
        self._max_episode_length = int(config.max_episode_length)
        self._sparse_reward_on_success = float(config.sparse_reward_on_success)
        self.display_image = bool(config.display_image)

        # Optional swap of the manipulated object's MJCF, resolved against this
        # package's own assets.  `milk_wide.xml` widens the stock carton from
        # 4.0 cm to 5.6 cm across, which is what makes it grippable by hand
        # without a dead-on approach.
        self._object_xml = None
        object_xml = config.get("object_xml")
        if object_xml:
            self._object_xml = _ASSET_DIR / str(object_xml)
            if not self._object_xml.is_file():
                raise FileNotFoundError(
                    f"object_xml {object_xml!r} not found at {self._object_xml}"
                )

        # Per-axis ceiling on the six Cartesian channels, or None for no limit.
        # Given as [translation, rotation] and broadcast to the two triplets.
        self._action_limit = None
        action_limit = config.get("action_limit")
        if action_limit is not None:
            limit = as_array(action_limit, "action_limit", ((2,),))
            if np.any(limit <= 0.0) or np.any(limit > 1.0):
                raise ValueError(
                    f"action_limit entries must lie in (0, 1], got {list(limit)}"
                )
            self._action_limit = np.repeat(limit, 3).astype(np.float32)

        # Small negative reward for commanding the gripper against its current
        # state, i.e. an actual open<->close transition rather than a command
        # that agrees with where the fingers already are.  The paper (Sec. 4.1)
        # asks for "a small negative penalty for gripper actions to discourage
        # the policy from operating its grippers unnecessarily" without giving a
        # value; -0.05 is upstream HIL-SERL's.
        #
        # Emitted through `info`, never folded into `reward`: it is consumed only
        # by the grasp critic's Bellman target, so the arm's SAC critic does not
        # pay for the gripper's decisions.  That split follows upstream's
        # `sac_hybrid_single.py`; the paper's own two-MDP formulation shares one
        # `r` between them and would imply folding it into both.
        self._grasp_penalty = float(config.get("grasp_penalty") or 0.0)
        if self._grasp_penalty > 0.0:
            raise ValueError(
                f"grasp_penalty must be <= 0 (it is a penalty), got "
                f"{self._grasp_penalty}"
            )

        # Optional relocation of the `agentview` camera.  The stock one sits at
        # (1.0, 0, 1.75), i.e. directly opposite the robot looking back at it,
        # which mirrors the horizontal plane: pushing a teleop device right
        # drives the arm left on screen.  Moving the camera behind the robot
        # fixes that at the source, without touching action signs -- which would
        # otherwise desynchronise hand-recorded demos from scripted ones.
        self._agentview_pos = None
        self._agentview_target = None
        if config.get("agentview_pos") is not None:
            self._agentview_pos = as_array(
                config.agentview_pos, "agentview_pos", ((3,),)
            )
            self._agentview_target = as_array(
                config.agentview_lookat, "agentview_lookat", ((3,),)
            )

        # Roll of the wrist camera about its own optical axis, in degrees.  The
        # stock mount is rotated 90 degrees relative to the scene camera, so a
        # motion that reads as left-right in the third-person view reads as
        # up-down on the wrist -- an operator watching the wrist feed sees their
        # sideways push move the image the wrong way entirely.  Rolling the
        # camera aligns the two feeds; it changes only the viewpoint, never the
        # dynamics.
        self._wrist_camera_roll = float(config.get("wrist_camera_roll") or 0.0)

        # Where the carton starts, relative to bin1's centre, in metres.  Unset
        # keeps robosuite's own scatter over the whole bin; see
        # `_apply_object_init` for why this is applied after construction.
        self._object_init_pos = None
        if config.get("object_init_pos") is not None:
            self._object_init_pos = as_array(
                config.object_init_pos, "object_init_pos", ((2,),)
            )
        # Yaw in radians.  None keeps the stock full-circle randomisation.
        raw_rot = config.get("object_init_rot")
        self._object_init_rot = None if raw_rot is None else float(raw_rot)
        # Half-widths, so 0 pins the pose exactly and a small value keeps enough
        # variety that the policy cannot simply memorise one approach.
        self._object_init_jitter = float(config.get("object_init_jitter") or 0.0)
        self._object_init_rot_jitter = float(
            config.get("object_init_rot_jitter") or 0.0
        )
        if self._object_init_jitter < 0.0 or self._object_init_rot_jitter < 0.0:
            raise ValueError(
                "object_init_jitter and object_init_rot_jitter are half-widths "
                "and cannot be negative"
            )

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
                        key: gym.spaces.Box(-np.inf, np.inf, shape=(dim,))
                        for key, dim in _STATE_DIMS.items()
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0, 255, shape=(height, width, 3), dtype=np.uint8
                        )
                        for key in (_AGENT_CAMERA, _WRIST_CAMERA)
                    }
                ),
            }
        )
        # 7-dim: xyz delta, rotvec delta, gripper -- the same layout as
        # `pick_cube_sim`, and already what robosuite's OSC_POSE + GRIP expects.
        self.action_space = gym.spaces.Box(
            -np.ones((7,), dtype=np.float32),
            np.ones((7,), dtype=np.float32),
            dtype=np.float32,
        )

        # Latched, not integrated -- see the module docstring.  Episodes start
        # open, because the arm resets above an object it has not grasped yet.
        self._gripper_latch = _ROBOSUITE_OPEN

        self._env = None
        self._display_open = False
        self._goal_pos = np.zeros(3)
        if self.fake_env:
            # The learner only ever needs the spaces above; returning here keeps
            # robosuite and its MuJoCo scene out of its process entirely.
            return

        self._env = self._make_env()
        self._apply_agentview_pose()
        self._object_name = self._env.objects[self._env.object_id].name
        self._apply_object_init()

    # ------------------------------------------------------------------ setup

    def _apply_object_init(self) -> None:
        """Narrow or pin where the carton starts, per config.

        Stock robosuite scatters it over the whole of bin1 -- measured here,
        20 cm x 30 cm of travel and a full 360 degrees of yaw -- which is a wide
        state distribution to learn a grasp over.  Pinning it makes the task
        converge far faster; a small range keeps some generalisation.

        Written onto the sampler after construction rather than passed in:
        `PickPlace` takes no `placement_initializer` argument, and
        `_load_model` calls `_get_placement_initializer` unconditionally, so
        anything handed to the constructor would be overwritten.  The sampler is
        built once and re-`sample()`d on every reset, so editing it here holds
        for the life of the env.

        Only the task's own object is pinned.  Under `single_object_mode=2` the
        other three still share bin1 and are still placed by the stock sampler,
        which rejects overlapping draws -- pinning all four to one point would
        make that rejection unsatisfiable and raise `RandomizationError`.
        """
        if self._object_init_pos is None and self._object_init_rot is None:
            return

        from robosuite.utils.placement_samplers import UniformRandomSampler

        initializer = self._env.placement_initializer
        samplers = getattr(initializer, "samplers", None)
        if not samplers:
            raise RuntimeError(
                "Could not reach robosuite's placement sampler to apply "
                "object_init_pos/object_init_rot. This env pins the carton by "
                "editing that sampler, so a robosuite version that builds it "
                "differently needs this method updated."
            )
        # The first sampler covers the graspable objects in bin1; the later ones
        # place the visual goal markers and must be left alone.
        name, bin_sampler = next(iter(samplers.items()))
        target = self._env.objects[self._env.object_id]

        pos = self._object_init_pos
        x, y = (0.0, 0.0) if pos is None else (float(pos[0]), float(pos[1]))
        jitter = self._object_init_jitter
        rotation = bin_sampler.rotation
        if self._object_init_rot is not None:
            # A scalar is a fixed angle; a 2-tuple is a range.
            rotation = (
                float(self._object_init_rot)
                if self._object_init_rot_jitter <= 0.0
                else (
                    self._object_init_rot - self._object_init_rot_jitter,
                    self._object_init_rot + self._object_init_rot_jitter,
                )
            )

        pinned = UniformRandomSampler(
            name="PinnedTargetSampler",
            mujoco_objects=[target],
            x_range=[x - jitter, x + jitter],
            y_range=[y - jitter, y + jitter],
            rotation=rotation,
            rotation_axis="z",
            # The stock sampler insets each range by the object's radius to keep
            # it clear of the bin walls, which turns a pinned (zero-width) range
            # negative and makes `np.random.uniform` raise `high - low < 0`.
            # The pose here is a specific point rather than a region to fit the
            # carton into, so staying inside the bin is the caller's business --
            # hence the measured bounds quoted in the config.
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self._env.bin1_pos,
            z_offset=bin_sampler.z_offset,
        )
        # Placed first so the other three are then sampled around it, rather
        # than it having to avoid wherever they happened to land.
        bin_sampler.mujoco_objects = [
            obj for obj in bin_sampler.mujoco_objects if obj is not target
        ]
        initializer.samplers = {
            "PinnedTargetSampler": pinned,
            name: bin_sampler,
            **{k: v for k, v in samplers.items() if k != name},
        }

    @contextmanager
    def _object_xml_override(self):
        """Point ``MilkObject`` at this repo's wider carton for the duration.

        robosuite builds its objects from hard-coded asset paths inside
        `xml_objects.py`, with no hook to substitute one, so the swap has to
        happen by patching that class while the scene is constructed.  The patch
        is scoped to `robosuite.make` and reverted in the `finally`, which
        matters because the class is module-global: leaving it in place would
        silently change every other robosuite env in the same process.

        Editing the installed package instead would work until the next
        `uv sync` wiped it, and would not travel with the repo.
        """
        if self._object_xml is None:
            yield
            return

        from robosuite.models.objects import xml_objects

        original = xml_objects.MilkObject
        path = str(self._object_xml)

        class _OverriddenMilkObject(xml_objects.MujocoXMLObject):
            def __init__(self, name):
                super().__init__(
                    path,
                    name=name,
                    joints=[dict(type="free", damping="0.0005")],
                    obj_type="all",
                    duplicate_collision_geoms=True,
                )

        xml_objects.MilkObject = _OverriddenMilkObject
        # `pick_place` imported the name directly, so rebinding the module
        # attribute alone would not reach it.
        from robosuite.environments.manipulation import pick_place

        pick_place.MilkObject = _OverriddenMilkObject
        try:
            yield
        finally:
            xml_objects.MilkObject = original
            pick_place.MilkObject = original

    def _make_env(self):
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(
            controller=self._controller, robot=self._robot
        )
        height, width = self._image_size
        with self._object_xml_override():
            return robosuite.make(
                self._env_name,
                robots=self._robot,
                controller_configs=controller_config,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=[_AGENT_CAMERA, _WRIST_CAMERA],
                camera_heights=height,
                camera_widths=width,
                control_freq=self._control_freq,
                reward_shaping=False,
                # The time limit is owned here; see the module docstring.
                ignore_done=True,
                # A hard reset reloads the model and sim on every episode.
                # Object placement is re-randomised either way, in
                # `_reset_internal`.
                hard_reset=False,
                seed=self._seed,
            )

    # --------------------------------------------------------------- gym API

    def _apply_agentview_pose(self) -> None:
        """Move the `agentview` camera and roll the wrist one, per config.

        Written into the loaded ``MjModel`` rather than the MJCF, so it survives
        `robosuite.make` without rebuilding the scene.  ``sim.forward()`` is what
        makes the new pose visible to the renderer -- without it the camera
        matrices are stale and frames come back from the old viewpoint.
        """
        sim = self._env.sim
        if self._agentview_pos is not None:
            camera_id = sim.model.camera_name2id(_AGENT_CAMERA)
            sim.model.cam_pos[camera_id] = self._agentview_pos
            sim.model.cam_quat[camera_id] = _look_at_quat(
                self._agentview_pos, self._agentview_target
            )
        if self._wrist_camera_roll:
            from scipy.spatial.transform import Rotation

            camera_id = sim.model.camera_name2id(_WRIST_CAMERA)
            w, x, y, z = sim.model.cam_quat[camera_id]
            rolled = Rotation.from_quat([x, y, z, w]) * Rotation.from_euler(
                "z", self._wrist_camera_roll, degrees=True
            )
            x, y, z, w = rolled.as_quat()
            sim.model.cam_quat[camera_id] = np.array([w, x, y, z])
        sim.forward()

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.cur_episode_length = 0
        self._gripper_latch = _ROBOSUITE_OPEN
        if self.fake_env:
            return self._zero_observation(), {}

        if seed is None and not self._seeded:
            seed = self._seed
        self._seeded = True
        if seed is not None:
            # robosuite seeds only at construction, and its placement samplers
            # hold a reference to `env.rng` rather than reseeding themselves.
            # Overwriting the generator's state in place is therefore the one
            # way to make `reset(seed=...)` reproducible -- which evaluation
            # relies on, since it seeds every episode separately.
            self._env.rng.bit_generator.state = np.random.default_rng(
                seed
            ).bit_generator.state

        obs = self._env.reset()
        self._goal_pos = np.asarray(
            self._env.target_bin_placements[self._env.object_id], dtype=np.float64
        )
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
        # Applied here rather than in the actor loop so that every consumer --
        # online rollouts, evaluation, and a replayed demonstration -- sees the
        # same limits without any of them having to know about it.  The action
        # space itself stays [-1, 1]: shrinking it instead would silently
        # invalidate the demo files and every stored checkpoint.
        action = self._clip_action_magnitude(action)
        self.cur_episode_length += 1

        if self.fake_env:
            return (
                self._zero_observation(),
                0.0,
                False,
                False,
                {"succeed": False, "grasp_penalty": 0.0},
            )

        # Judged on whether the command would *flip the latch*, not on where the
        # fingers currently sit.  Upstream HIL-SERL thresholds the finger
        # position (open > 0.9) because a real Franka gripper is fast enough to
        # be effectively binary; robosuite's PandaGripper integrates at
        # speed=0.2, so it spends tens of steps mid-travel.  Measured here, the
        # opening runs 0.013 to 0.992 and sits at 0.52 right after a reset, which
        # breaks a position threshold in both directions at once: a close command
        # at reset is not penalised (0.52 > 0.9 is false) while a repeated open
        # command is penalised on every step of the travel.
        #
        # The latch is the gripper's actual commanded state, so comparing against
        # it charges exactly once per open<->close transition and never for a
        # command that merely repeats the current one.
        if action[6] < _GRIPPER_CLOSE_THRESHOLD:
            commanded_latch = _ROBOSUITE_CLOSE
        elif action[6] > _GRIPPER_OPEN_THRESHOLD:
            commanded_latch = _ROBOSUITE_OPEN
        else:
            commanded_latch = self._gripper_latch
        grasp_penalty = (
            self._grasp_penalty if commanded_latch != self._gripper_latch else 0.0
        )

        # Latch: only a decisive command changes the gripper, so the grasp holds
        # through the many steps where the operator is pushing translation only.
        # The latched value is then re-sent every step, which is what an
        # integrating actuator needs to reach and hold its endpoint.
        self._gripper_latch = commanded_latch

        obs, _, _, _ = self._env.step(
            np.concatenate([action[:6], [self._gripper_latch]])
        )

        succeed = bool(self._env._check_success())
        # Recomputed rather than taken from robosuite's reward so that one config
        # knob drives it, matching the ManiSkill env.
        reward = self._sparse_reward_on_success if succeed else 0.0
        terminated = succeed
        truncated = (
            self.cur_episode_length >= self._max_episode_length and not terminated
        )

        observation = self._extract_observation(obs)
        self._maybe_display(observation["images"])
        info = {
            "succeed": succeed,
            "grasped": self._is_grasped(),
            "object_to_goal": float(
                np.linalg.norm(observation["state"]["object_to_goal"])
            ),
            # Consumed only by the grasp critic's target; never folded into
            # `reward`.  See the constructor.
            "grasp_penalty": grasp_penalty,
        }
        return observation, reward, terminated, truncated, info

    def _clip_action_magnitude(self, action: np.ndarray) -> np.ndarray:
        """Hold the policy to the same envelope the operator drove in.

        A SpaceMouse geared for controllable hand teleoperation reaches only a
        fraction of the action space -- measured over 20 recorded episodes here,
        translation peaked at 0.252 and rotation at 0.107 -- while an untrained
        SAC policy samples |a| ~= 0.66 by default.  Left alone, the arm lurches
        several times faster under the policy than under the human, which makes
        an intervention feel like grabbing a different robot and gives the critic
        a swathe of actions no demonstration ever covers.

        Clipped per axis rather than by vector norm, so a diagonal push keeps its
        direction instead of being scaled back towards an axis.  The gripper
        channel is untouched: it is a latch, not a velocity, and rescaling it
        would push a decisive command back under the +-0.5 threshold.
        """
        if self._action_limit is None:
            return action
        action = action.copy()
        np.clip(
            action[:6], -self._action_limit, self._action_limit, out=action[:6]
        )
        return action

    def render_camera(
        self,
        camera_name: str,
        width: int = 256,
        height: int = 256,
        mode: Literal["rgb_array"] = "rgb_array",
    ) -> np.ndarray:
        """Render an extra camera without adding it to policy observations.

        Signature matches ``MujocoGymEnv.render_camera`` so ``train_serl.py``'s
        evaluation-video path works unchanged.  Note this scene has no ``front``
        camera -- the flag's default -- so pass
        ``--eval_video_main_camera=agentview``.
        """
        if mode != "rgb_array":
            raise ValueError(
                f"RobosuitePickPlaceGymEnv only renders rgb_array, got {mode!r}"
            )
        if self.fake_env or self._env is None:
            return np.zeros((height, width, 3), dtype=np.uint8)

        available = list(self._env.sim.model.camera_names)
        if camera_name not in available:
            raise ValueError(
                f"Unknown render camera {camera_name!r}. Available: {sorted(available)}."
                "  Pass --eval_video_main_camera=agentview."
            )
        frame = self._env.sim.render(
            camera_name=camera_name, width=width, height=height
        )
        return np.ascontiguousarray(frame[::-1, :, :3], dtype=np.uint8)

    def close(self) -> None:
        env, self._env = self._env, None
        if env is not None:
            env.close()
        if self._display_open:
            cv2.destroyWindow(_DISPLAY_WINDOW)
            self._display_open = False

    # -------------------------------------------------------------- internals

    def _is_grasped(self) -> bool:
        target = self._env.objects[self._env.object_id]
        return bool(
            self._env._check_grasp(
                gripper=self._env.robots[0].gripper,
                object_geoms=target.contact_geoms,
            )
        )

    def _extract_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        prefix = self._object_name
        robot = self._env.robots[0]

        tcp_pose = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                np.asarray(obs["robot0_eef_quat_site"], dtype=np.float64),
            ]
        )
        object_pos = np.asarray(obs[f"{prefix}_pos"], dtype=np.float64)
        object_pose = np.concatenate(
            [object_pos, np.asarray(obs[f"{prefix}_quat"], dtype=np.float64)]
        )

        gripper_opening = float(
            np.clip(float(obs["robot0_gripper_qpos"][0]) / _FINGER_QPOS_OPEN, 0.0, 1.0)
        )

        # `ee_force`/`ee_torque` are dicts keyed by arm; `_hand_vel`/`_hand_ang_vel`
        # are the linear and angular halves of the same twist.
        state = {
            "tcp_pose": tcp_pose,
            "tcp_vel": np.concatenate(
                [
                    np.asarray(robot._hand_vel[_ARM_NAME], dtype=np.float64).reshape(-1),
                    np.asarray(
                        robot._hand_ang_vel[_ARM_NAME], dtype=np.float64
                    ).reshape(-1),
                ]
            ),
            "tcp_force": np.asarray(
                robot.ee_force[_ARM_NAME], dtype=np.float64
            ).reshape(-1),
            "tcp_torque": np.asarray(
                robot.ee_torque[_ARM_NAME], dtype=np.float64
            ).reshape(-1),
            "gripper_pose": np.asarray([gripper_opening], dtype=np.float64),
            "object_pose": object_pose,
            "object_to_goal": self._goal_pos - object_pos,
        }

        images = {}
        for key in (_AGENT_CAMERA, _WRIST_CAMERA):
            frame = np.asarray(obs[f"{key}_image"])
            # OpenGL convention: robosuite hands back a vertically flipped frame.
            images[key] = np.ascontiguousarray(frame[::-1], dtype=np.uint8)

        return {"state": state, "images": images}

    def _zero_observation(self) -> Dict[str, Any]:
        # Built from _STATE_DIMS rather than from `self.observation_space`:
        # Quat2RotvecWrapper rewrites the `tcp_pose` entry of that space *in
        # place*, to 6 dims, so deriving shapes from it yields whatever the
        # outer wrappers have already done to it.
        state = {
            key: np.zeros(dim, dtype=np.float64) for key, dim in _STATE_DIMS.items()
        }
        # An all-zero quaternion is not a rotation, and RelativeFrame builds a
        # transform out of `tcp_pose` on the learner's very first reset -- where
        # scipy rejects it outright.  The pose itself is meaningless here; only
        # its validity matters.
        state["tcp_pose"][6] = 1.0
        state["object_pose"][6] = 1.0
        height, width = self._image_size
        return {
            "state": state,
            "images": {
                key: np.zeros((height, width, 3), dtype=np.uint8)
                for key in (_AGENT_CAMERA, _WRIST_CAMERA)
            },
        }

    def _maybe_display(self, images: Dict[str, np.ndarray]) -> None:
        if not self.display_image:
            return
        frame = np.concatenate([images[_AGENT_CAMERA], images[_WRIST_CAMERA]], axis=1)
        cv2.imshow(_DISPLAY_WINDOW, frame[..., ::-1])  # cv2 windows want BGR
        cv2.waitKey(1)
        self._display_open = True

    # Read by the scripted expert, which normalises its Cartesian deltas by the
    # controller's own output scale so that retuning the controller does not
    # silently break the expert.
    @property
    def position_action_scale(self) -> float:
        return float(
            self._env.robots[0].part_controllers[_ARM_NAME].output_max[0]
        )

    @property
    def goal_position(self) -> np.ndarray:
        return self._goal_pos.copy()
