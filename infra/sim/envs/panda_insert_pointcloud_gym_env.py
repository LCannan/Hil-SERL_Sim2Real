import mujoco
import fpsample
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig

from infra.utils.vision_util import (
    depth_to_point_cloud,
    PointCloudDisplayer,
)
from infra.utils.transformations import construct_homogeneous_matrix
from typing import Any, Literal, Tuple, Dict
from infra.sim.envs.mujoco_gym_env import GymRenderingSpec
from infra.sim.envs.panda_insert_gym_env import PandaPegInsertGymEnv


class PandaPegInsertDepthGymEnv(PandaPegInsertGymEnv):
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
        num_points: int = 512,
        min_world_z: float = 0.005,
        max_world_z: float = 1.2,
    ):
        if render_spec is None:
            render_spec = [
                GymRenderingSpec(camera_name="wrist1", mode="depth_array"),
                GymRenderingSpec(camera_name="wrist2", mode="depth_array"),
            ]
        self.num_points = int(num_points)
        self.min_world_z = float(min_world_z)
        self.max_world_z = float(max_world_z)

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
            render_mode=render_mode,
            config=config,
            fake_env=fake_env,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "point_cloud": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.num_points, 3),
                            dtype=np.float32,
                        )
                    }
                ),
            }
        )

        self.camera_intrinsics = {
            spec.camera_name: self._get_camera_intrinsics(
                spec.camera_name,
                spec.height,
                spec.width,
            )
            for spec in render_spec
        }

        self._pc_displayer = None

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset(seed=seed, **kwargs)
        point_cloud = self._get_pointcloud(obs)
        obs["images"] = {"point_cloud": point_cloud}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, reward, done, truncated, info = super().step(action)
        point_cloud = self._get_pointcloud(obs)

        if self.display_image and not self.fake_env and self._pc_displayer is None:
            self._pc_displayer = PointCloudDisplayer(points=point_cloud)
        if self._pc_displayer is not None:
            self._pc_displayer.display(points=point_cloud) 

        obs["images"] = {"point_cloud": point_cloud}
        
        return obs, reward, done, truncated, info

    def _get_camera_intrinsics(self, camera_name: str, height: int, width: int):
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self._model.cam_fovy[cam_id]
        fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        fx = fy
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        return fx, fy, cx, cy

    def _get_pointcloud(self, obs):
        # 1. reconstruct each depth camera and filter points in world coordinates
        world_clouds = []
        for spec in self._render_specs:
            depth = self._depth_buffer_to_meters(obs["images"][spec.camera_name])
            fx, fy, cx, cy = self.camera_intrinsics[spec.camera_name]
            camera_cloud = depth_to_point_cloud(depth, fx, fy, cx, cy)
            camera_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                spec.camera_name,
            )
            world_clouds.append(
                self._filter_point_cloud(camera_cloud, self._data, camera_id)
            )

        # 2. merge cameras and farthest-point sample a fixed-size observation
        merged = np.concatenate(world_clouds, axis=0).astype(np.float32)
        if len(merged) == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)
        merged = self._sample_fixed_size(merged, self.num_points)

        # 3. convert point cloud to tool frame
        T_world_tcp = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
        T_tcp_world = np.linalg.inv(T_world_tcp)
        ones = np.ones((merged.shape[0], 1), dtype=np.float32)
        pw_h = np.concatenate([merged.astype(np.float64), ones], axis=1)  # (N,4)
        pe_h = (T_tcp_world @ pw_h.T).T
        point_cloud_tcp = pe_h[:, :3].astype(np.float32)

        return point_cloud_tcp

    def _sample_fixed_size(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """FPS sample a fixed-size cloud and pad sparse clouds deterministically."""
        if len(points) == 0:
            return np.zeros((num_points, 3), dtype=np.float32)
        if len(points) >= num_points:
            indices = fpsample.fps_sampling(points[:, :3], num_points)
            return points[indices]

        repeats = int(np.ceil(num_points / len(points)))
        return np.tile(points, (repeats, 1))[:num_points]
    
    def _depth_buffer_to_meters(self, depth_buffer: np.ndarray) -> np.ndarray:
        near = self._model.vis.map.znear * self._model.stat.extent
        far = self._model.vis.map.zfar * self._model.stat.extent
        depth_m = near / (1.0 - depth_buffer * (1.0 - near / far))
        return depth_m
    
    def _filter_point_cloud(self, points_cam: np.ndarray, data, cam_id: int) -> np.ndarray:
        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
        # Back-projection above uses OpenCV-like camera coords:
        # x right, y down, z forward.
        # MuJoCo/OpenGL camera convention differs by a flip on y and z.
        cv_to_mj = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
        points_mj_cam = points_cam @ cv_to_mj.T
        points_world = points_mj_cam @ cam_rot.T + cam_pos

        z = points_world[:, 2]
        mask = (
            np.isfinite(points_world).all(axis=1)
            & (z >= self.min_world_z)
            & (z <= self.max_world_z)
        )
        points_world = points_world[mask]
        return points_world

    def close(self) -> None:
        if self._pc_displayer is not None:
            self._pc_displayer.close()
            self._pc_displayer = None
        super().close()
