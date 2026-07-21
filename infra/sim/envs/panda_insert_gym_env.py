from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
import mujoco
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from infra.sim.controllers.impedance import impedance_control
from infra.sim.envs.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from infra.utils.config_util import as_array

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "assets" / "panda_peg_insert.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))


class PandaPegInsertGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        config: DictConfig,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: list[GymRenderingSpec] | None = None,
        render_mode: Literal["rgb_array", "human"] | None = None,
        fake_env: bool = False,
    ):
        if config is None:
            raise ValueError("PandaPegInsertGymEnv requires an environment config")
        if render_spec is None:
            render_spec = [
                GymRenderingSpec(camera_name="wrist1"),
                GymRenderingSpec(camera_name="wrist2"),
            ]
        if render_mode is None:
            render_mode = "rgb_array" if fake_env else "human"

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
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.config = config
        self.fake_env = bool(fake_env)
        self._action_scale = as_array(config.action_scale, "action_scale", ((2,),))
        self._target_pose = as_array(config.target_pose, "target_pose", ((7,),))
        self._reward_threshold = as_array(config.reward_threshold, "reward_threshold", ((6,),))
        reset_pose = as_array(config.reset_pose, "reset_pose", ((7,),))
        abs_xyz_limit_low = as_array(config.abs_xyz_limit_low, "abs_xyz_limit_low", ((3,),))
        abs_xyz_limit_high = as_array(config.abs_xyz_limit_high, "abs_xyz_limit_high", ((3,),))

        self._random_reset = bool(config.random_reset)
        self._random_xy_range = float(config.random_xy_range)
        self._random_rx_range = float(config.random_rx_range)
        self._random_ry_range = float(config.random_ry_range)
        self._random_rz_range = float(config.random_rz_range)
        self._max_episode_length = int(config.max_episode_length)
        self.display_image = bool(config.display_image)
        if self._max_episode_length <= 0:
            raise ValueError("max_episode_length must be positive")

        self.resetpos = reset_pose.copy()

        self.render_mode = render_mode

        self.cur_episode_length = 0
        self._panda_dof_ids = np.asarray([self._model.joint(f"joint{i}").id for i in range(1, 8)])
        
        self._panda_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self._model.actuator("actuator8").id

        self._hand_site_id = self._model.site("hand_site").id
        self._end_effector_id = self._model.body("end_effector").mocapid.item()

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

        self.action_space = gym.spaces.Box(
            np.ones((6,), dtype=np.float32) * -1,
            np.ones((6,), dtype=np.float32),
        )

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self._random = np.random.RandomState(seed)
        self.cur_episode_length = 0
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)

        reset_pose = self.resetpos.copy()

        if self._random_reset:
            reset_pose[:2] += self.random_state.uniform(
                -self._random_xy_range, self._random_xy_range, (2,)
            )

            quat_reset = reset_pose[3:].copy()
            euler_delta = np.array([
                self.random_state.uniform(-self._random_rx_range, self._random_rx_range),
                self.random_state.uniform(-self._random_ry_range, self._random_ry_range),
                self.random_state.uniform(-self._random_rz_range, self._random_rz_range),
            ])
            reset_pose[3:] = (Rotation.from_euler("xyz", euler_delta) * Rotation.from_quat(quat_reset)).as_quat()

        reset_quat_scipy = reset_pose[3:7]  # [x, y, z, w]  
        reset_pose[3:7] = np.array([reset_quat_scipy[3], reset_quat_scipy[0], reset_quat_scipy[1], reset_quat_scipy[2]])  # [w, x, y, z]
        self._reset_arm_to_home(reset_pose)
        self._data.mocap_pos[self._end_effector_id] = reset_pose[:3]
        self._data.mocap_quat[self._end_effector_id] = reset_pose[3:7]
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        cur_pos = self._data.mocap_pos[self._end_effector_id]
        cur_quat_mujoco = self._data.mocap_quat[self._end_effector_id]
        cur_quat = [cur_quat_mujoco[1], cur_quat_mujoco[2], cur_quat_mujoco[3], cur_quat_mujoco[0]]

        next_pos = cur_pos + xyz_delta * self._action_scale[0]
        next_quat = (Rotation.from_rotvec(action[3:6] * self._action_scale[1])
                    * Rotation.from_quat(cur_quat)).as_quat()
        next_pos_quat_clip = self._clip_safety_box(np.concatenate([next_pos, next_quat]))
        next_quat_mujoco = [next_pos_quat_clip[6], next_pos_quat_clip[3], next_pos_quat_clip[4], next_pos_quat_clip[5]]

        self._data.mocap_pos[self._end_effector_id] = next_pos_quat_clip[:3]
        self._data.mocap_quat[self._end_effector_id] = next_quat_mujoco

        self.cur_episode_length += 1
        self._servo_Impedance_pose(self._data.mocap_pos[self._end_effector_id], self._data.mocap_quat[self._end_effector_id], num_steps=self._n_substeps)

        obs = self._compute_observation()
        reward = self._compute_reward()
        terminated = bool(reward)
        truncated = (
            (
                self.cur_episode_length >= self._max_episode_length
                or self.time_limit_exceeded()
            )
            and not terminated
        )

        if self.render_mode == "human":
            self._viewer.sync()
        
        return obs, int(reward), terminated, truncated, {"succeed": bool(reward)}

    def _reset_arm_to_home(self, tcp_pos=None):
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.qvel[self._panda_dof_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._servo_Impedance_pose(tcp_pos[:3], tcp_pos[3:7])

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        cur_pos = self._data.site_xpos[self._hand_site_id].copy()
        cur_xmat = self._data.site_xmat[self._hand_site_id].reshape((3, 3))
        cur_rot = Rotation.from_matrix(cur_xmat)
        obs["state"]["tcp_pose"] = np.concatenate([cur_pos, cur_rot.as_quat()])

        obs["state"]["tcp_vel"] = self._get_site_twist(self._model, self._data, self._hand_site_id) 
        obs["state"]["tcp_force"] = self._data.sensor("panda/end_effector_force").data
        obs["state"]["tcp_torque"] = self._data.sensor("panda/end_effector_torque").data

        obs["images"] = {
            spec.camera_name: frame
            for spec, frame in zip(self._render_specs, self.render())
        }

        return obs

    def _compute_reward(self) -> bool:
        cur_pos = self._data.site_xpos[self._hand_site_id].copy()
        cur_xmat = self._data.site_xmat[self._hand_site_id].reshape((3, 3))
        
        cur_rot = Rotation.from_matrix(cur_xmat)
        position_error = np.abs(cur_pos - self._target_pose[:3])
        rotation_error = np.abs(
            (
                cur_rot
                * Rotation.from_quat(self._target_pose[3:]).inv()
            ).as_rotvec()
        )
        delta = np.concatenate([position_error, rotation_error])
        
        return bool(np.all(delta < self._reward_threshold))
    
    def _servo_Impedance_pose(self, target_pos, target_quat, num_steps=2000):
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
            mujoco.mj_step(self._model, self._data)
            self._data.ctrl[self._gripper_ctrl_id] = 0.0

        mujoco.mj_forward(self._model, self._data)

    def _get_site_twist(self,model, data, site_id, local=False):
        vel = np.zeros((6, 1), dtype=np.float64)  
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel, local)
        return vel.reshape(6,)

    def _clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        
        target_rotation = Rotation.from_quat(self._target_pose[3:])
        delta_R = Rotation.from_quat(pose[3:]) * target_rotation.inv()
        delta_euler = delta_R.as_euler("xyz")
        delta_euler = np.clip(
            delta_euler,
            [-self._random_rx_range, -self._random_ry_range, -self._random_rz_range],
            [self._random_rx_range, self._random_ry_range, self._random_rz_range]
        )
        pose[3:] = (
            Rotation.from_euler("xyz", delta_euler) * target_rotation
        ).as_quat()

        return pose
