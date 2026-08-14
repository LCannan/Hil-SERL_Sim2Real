"""Pick-and-place a cube to a randomized 3D goal, in MuJoCo.

Shares the impedance-servo scheme of :mod:`panda_insert_gym_env` -- actions are
deltas on a mocap setpoint the arm chases, not joint torques -- but differs from
every other Cartesian task here in three ways that the wrapper stack and the
configs both depend on:

- **The action is 7-dim, not 6.**  The seventh is the gripper.  This is the only
  environment in the repo with one; ``insert_sim`` welds its peg to the finger.
  ``RelativeFrame`` and ``Quat2RotvecWrapper`` both operate on ``action[:6]``
  only, so they still apply unchanged (verified, not assumed).
- **The gripper command is latched, not integrated.**  ``action[6] < -0.5``
  closes, ``> 0.5`` opens, anything between holds the current state.  A grasp has
  to survive the many steps of a lift during which a human is pushing only the
  translation axes and the gripper channel reads ~0, and an integrating channel
  would drift open exactly then.
- **Success needs a held cube, not just a placed one.**  Contact with both
  fingers is required, so dropping the cube through the goal sphere on a
  ballistic arc does not count.

The MuJoCo gripper actuator is inverted with respect to the usual convention:
``actuator8`` is a position servo on the ``split`` tendon with ``ctrlrange
0..255`` where **0 is fully closed and 255 fully open** (measured).  Hence
``_GRIPPER_CLOSED_CTRL``/``_GRIPPER_OPEN_CTRL`` rather than a bare 0/255 pair at
the call site.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from infra.sim.controllers.impedance import impedance_control
from infra.sim.envs.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from infra.utils.config_util import as_array

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "assets" / "panda_pick_cube_sim.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))

# See the module docstring: 0 closes the fingers, 255 opens them.
_GRIPPER_CLOSED_CTRL = 0.0
_GRIPPER_OPEN_CTRL = 255.0

_CUBE_HALF_SIZE = 0.02


class PandaPickCubeGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        config: DictConfig,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: list[GymRenderingSpec] | None = None,
        render_mode: Literal["rgb_array", "human"] | None = None,
        fake_env: bool = False,
    ):
        if config is None:
            raise ValueError("PandaPickCubeGymEnv requires an environment config")
        if render_spec is None:
            render_spec = [
                GymRenderingSpec(camera_name="wrist1"),
                GymRenderingSpec(camera_name="wrist2"),
            ]
        if render_mode is None:
            # Off-screen only by default, matching the insert task: opening an
            # interactive viewer makes actors fail on headless machines.
            render_mode = "rgb_array"

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
            render_mode=render_mode,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.config = config
        self.fake_env = bool(fake_env)
        self._action_scale = as_array(config.action_scale, "action_scale", ((2,),))
        self._reset_pose = as_array(config.reset_pose, "reset_pose", ((7,),))
        abs_xyz_limit_low = as_array(
            config.abs_xyz_limit_low, "abs_xyz_limit_low", ((3,),)
        )
        abs_xyz_limit_high = as_array(
            config.abs_xyz_limit_high, "abs_xyz_limit_high", ((3,),)
        )
        self._cube_xy_center = as_array(
            config.cube_xy_center, "cube_xy_center", ((2,),)
        )
        self._goal_xyz_center = as_array(
            config.goal_xyz_center, "goal_xyz_center", ((3,),)
        )
        self._goal_xyz_range = as_array(
            config.goal_xyz_range, "goal_xyz_range", ((3,),)
        )

        self._cube_xy_range = float(config.cube_xy_range)
        self._goal_threshold = float(config.goal_threshold)
        self._random_reset = bool(config.random_reset)
        self._random_rz_range = float(config.random_rz_range)
        self._max_episode_length = int(config.max_episode_length)
        self.display_image = bool(config.display_image)
        if self._max_episode_length <= 0:
            raise ValueError("max_episode_length must be positive")
        if self._goal_threshold <= 0.0:
            raise ValueError("goal_threshold must be positive")

        self.render_mode = render_mode
        self.cur_episode_length = 0

        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("actuator8").id
        self._hand_site_id = self._model.site("hand_site").id
        self._end_effector_id = self._model.body("end_effector").mocapid.item()
        self._goal_marker_id = self._model.body("goal_marker").mocapid.item()

        self._cube_qpos_adr = self._model.joint("cube").qposadr[0]
        self._cube_qvel_adr = self._model.joint("cube").dofadr[0]
        self._cube_geom_id = self._model.geom("cube").id
        # Both fingers carry several collision geoms (pads plus the mesh), so
        # grasp detection has to test membership in a set rather than compare a
        # single id per side.
        self._left_finger_geom_ids = self._finger_geom_ids("left_finger")
        self._right_finger_geom_ids = self._finger_geom_ids("right_finger")

        # Latched, not integrated -- see the module docstring.  Episodes start
        # open, because the arm resets above a cube it has not grasped yet.
        self._gripper_ctrl = _GRIPPER_OPEN_CTRL

        self._goal_pos = self._goal_xyz_center.copy()

        self.xyz_bounding_box = gym.spaces.Box(
            abs_xyz_limit_low,
            abs_xyz_limit_high,
            dtype=np.float64,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        # Normalised finger opening, so the policy can tell a
                        # closed-and-holding hand from a closed-and-empty one
                        # when combined with the force reading.
                        "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                        # Privileged, and deliberately so: this is a sim task
                        # whose point is validating the HIL plumbing, not
                        # learning cube pose from pixels alone.
                        "cube_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "cube_to_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        spec.camera_name: gym.spaces.Box(
                            0,
                            255,
                            shape=(spec.height, spec.width, 3),
                            dtype=np.uint8,
                        )
                        for spec in render_spec
                    }
                ),
            }
        )

        # 7-dim: xyz delta, rotvec delta, gripper.
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

    def _finger_geom_ids(self, body_name: str) -> frozenset:
        body_id = self._model.body(body_name).id
        return frozenset(
            geom_id
            for geom_id in range(self._model.ngeom)
            if self._model.geom(geom_id).bodyid == body_id
        )

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self._random = np.random.RandomState(seed)
        self.cur_episode_length = 0
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)

        reset_pose = self._reset_pose.copy()
        cube_xy = self._cube_xy_center.copy()
        goal_pos = self._goal_xyz_center.copy()

        if self._random_reset:
            cube_xy += self.random_state.uniform(
                -self._cube_xy_range, self._cube_xy_range, (2,)
            )
            goal_pos += self.random_state.uniform(
                -self._goal_xyz_range, self._goal_xyz_range
            )
            # Only yaw is randomised.  Roll/pitch noise on a top-down grasp tips
            # the fingers off the cube's flat faces, and the impedance servo has
            # to fight it for the whole episode.
            quat_reset = reset_pose[3:].copy()
            euler_delta = np.array(
                [
                    0.0,
                    0.0,
                    self.random_state.uniform(
                        -self._random_rz_range, self._random_rz_range
                    ),
                ]
            )
            reset_pose[3:] = (
                Rotation.from_euler("xyz", euler_delta) * Rotation.from_quat(quat_reset)
            ).as_quat()

        # The arm resets above the cube rather than at a fixed pose: a human (or
        # a policy) that has to first hunt for the cube spends the whole episode
        # travelling, and the xyz box below is only 20 cm wide.
        reset_pose[:2] = cube_xy

        self._goal_pos = goal_pos
        self._data.mocap_pos[self._goal_marker_id] = goal_pos

        cube_qpos = np.zeros(7)
        cube_qpos[:2] = cube_xy
        cube_qpos[2] = _CUBE_HALF_SIZE
        cube_qpos[3] = 1.0  # wxyz identity
        self._data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_qpos
        self._data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6] = 0.0

        reset_quat_scipy = reset_pose[3:7]  # [x, y, z, w]
        reset_pose[3:7] = np.array(
            [
                reset_quat_scipy[3],
                reset_quat_scipy[0],
                reset_quat_scipy[1],
                reset_quat_scipy[2],
            ]
        )  # [w, x, y, z]

        self._gripper_ctrl = _GRIPPER_OPEN_CTRL
        self._reset_arm_to_home(reset_pose)
        self._data.mocap_pos[self._end_effector_id] = reset_pose[:3]
        self._data.mocap_quat[self._end_effector_id] = reset_pose[3:7]
        mujoco.mj_forward(self._model, self._data)

        return self._compute_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        cur_pos = self._data.mocap_pos[self._end_effector_id]
        cur_quat_mujoco = self._data.mocap_quat[self._end_effector_id]
        cur_quat = [
            cur_quat_mujoco[1],
            cur_quat_mujoco[2],
            cur_quat_mujoco[3],
            cur_quat_mujoco[0],
        ]

        next_pos = cur_pos + action[:3] * self._action_scale[0]
        next_quat = (
            Rotation.from_rotvec(action[3:6] * self._action_scale[1])
            * Rotation.from_quat(cur_quat)
        ).as_quat()
        next_pose = self._clip_safety_box(np.concatenate([next_pos, next_quat]))
        next_quat_mujoco = [
            next_pose[6],
            next_pose[3],
            next_pose[4],
            next_pose[5],
        ]

        self._data.mocap_pos[self._end_effector_id] = next_pose[:3]
        self._data.mocap_quat[self._end_effector_id] = next_quat_mujoco

        # Latch: only a decisive command changes the gripper, so the grasp holds
        # through the many steps where the operator is pushing translation only.
        if action[6] < -0.5:
            self._gripper_ctrl = _GRIPPER_CLOSED_CTRL
        elif action[6] > 0.5:
            self._gripper_ctrl = _GRIPPER_OPEN_CTRL

        self.cur_episode_length += 1
        self._servo_impedance_pose(
            self._data.mocap_pos[self._end_effector_id],
            self._data.mocap_quat[self._end_effector_id],
            num_steps=self._n_substeps,
        )

        obs = self._compute_observation()
        reward = self._compute_reward()
        terminated = bool(reward)
        truncated = (
            self.cur_episode_length >= self._max_episode_length
            or self.time_limit_exceeded()
        ) and not terminated

        if self.render_mode == "human":
            self._viewer.sync()

        info = {
            "succeed": bool(reward),
            "grasped": bool(self._is_grasped()),
            "cube_to_goal": float(
                np.linalg.norm(self._cube_position() - self._goal_pos)
            ),
        }
        return obs, int(reward), terminated, truncated, info

    def _reset_arm_to_home(self, tcp_pose=None):
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qvel[self._panda_dof_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._servo_impedance_pose(tcp_pose[:3], tcp_pose[3:7])

    def _cube_position(self) -> np.ndarray:
        return self._data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3].copy()

    def _cube_pose(self) -> np.ndarray:
        cube_qpos = self._data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7]
        quat_wxyz = cube_qpos[3:7]
        return np.concatenate(
            [
                cube_qpos[:3],
                # MuJoCo stores wxyz; every pose in this project is xyzw.
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            ]
        )

    def _is_grasped(self) -> bool:
        """True when both fingers are in contact with the cube.

        Contact rather than a finger-width threshold: the fingers close to
        roughly the cube's width whether or not the cube is actually between
        them, so width alone reports a successful grasp on thin air.
        """
        left = right = False
        for contact_id in range(self._data.ncon):
            contact = self._data.contact[contact_id]
            geom1, geom2 = contact.geom1, contact.geom2
            if self._cube_geom_id == geom1:
                other = geom2
            elif self._cube_geom_id == geom2:
                other = geom1
            else:
                continue
            if other in self._left_finger_geom_ids:
                left = True
            elif other in self._right_finger_geom_ids:
                right = True
            if left and right:
                return True
        return False

    def _compute_observation(self) -> dict:
        obs = {"state": {}}

        cur_pos = self._data.site_xpos[self._hand_site_id].copy()
        cur_xmat = self._data.site_xmat[self._hand_site_id].reshape((3, 3))
        cur_rot = Rotation.from_matrix(cur_xmat)
        obs["state"]["tcp_pose"] = np.concatenate([cur_pos, cur_rot.as_quat()])
        obs["state"]["tcp_vel"] = self._get_site_twist(
            self._model, self._data, self._hand_site_id
        )
        obs["state"]["tcp_force"] = self._data.sensor("panda/end_effector_force").data
        obs["state"]["tcp_torque"] = self._data.sensor("panda/end_effector_torque").data

        # Tendon length spans 0..0.04 per finger; report it normalised so the
        # scale matches the other state entries.
        finger_qpos = self._data.qpos[self._model.joint("finger_joint1").qposadr[0]]
        obs["state"]["gripper_pose"] = np.asarray(
            [np.clip(finger_qpos / 0.04, 0.0, 1.0)], dtype=np.float64
        )

        cube_pose = self._cube_pose()
        obs["state"]["cube_pose"] = cube_pose
        obs["state"]["cube_to_goal"] = self._goal_pos - cube_pose[:3]

        obs["images"] = {
            spec.camera_name: frame
            for spec, frame in zip(self._render_specs, self.render())
        }
        return obs

    def _compute_reward(self) -> bool:
        """Sparse: the cube is at the goal *and* still held.

        Requiring the grasp is what stops a thrown cube from scoring as it flies
        through the goal sphere.
        """
        if not self._is_grasped():
            return False
        distance = float(np.linalg.norm(self._cube_position() - self._goal_pos))
        return distance < self._goal_threshold

    def _servo_impedance_pose(self, target_pos, target_quat, num_steps=2000):
        for _ in range(num_steps):
            tau = impedance_control(
                model=self._model,
                data=self._data,
                site_id=self._hand_site_id,
                dof_ids=self._panda_dof_ids,
                pos=target_pos,
                ori=target_quat,
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            # Written before the step, unlike the insert env: the gripper here
            # carries a load, and applying last iteration's command would let the
            # fingers relax for one substep on the step a grasp begins.
            self._data.ctrl[self._gripper_ctrl_id] = self._gripper_ctrl
            mujoco.mj_step(self._model, self._data)

        mujoco.mj_forward(self._model, self._data)

    def _get_site_twist(self, model, data, site_id, local=False):
        vel = np.zeros((6, 1), dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel, local
        )
        return vel.reshape(6)

    def _clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )

        # Orientation is clipped against the *reset* pose, not a task target:
        # unlike insertion, picking has no goal orientation, and the constraint
        # that matters is keeping the hand pointing down at the cube.
        reference = Rotation.from_quat(self._reset_pose[3:])
        delta_euler = (Rotation.from_quat(pose[3:]) * reference.inv()).as_euler("xyz")
        delta_euler = np.clip(
            delta_euler,
            [0.0, 0.0, -self._random_rz_range],
            [0.0, 0.0, self._random_rz_range],
        )
        pose[3:] = (Rotation.from_euler("xyz", delta_euler) * reference).as_quat()
        return pose
