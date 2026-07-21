import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import queue
from collections import OrderedDict
from collections.abc import Mapping
from typing import Dict
from scipy.spatial.transform import Slerp

from omegaconf import DictConfig

from infra.hardware.camera.video_capture import VideoCapture
from infra.hardware.camera.rs_capture import RSCapture
from infra.hardware.robot.franka_client import FrankaApiClient
from infra.utils.config_util import as_array, as_dict



class FrankaEnv(gym.Env):
    def __init__(
        self,
        config: DictConfig,
        fake_env=False,
    ):
        if config is None:
            raise ValueError("FrankaEnv requires an environment config")

        self.config = config
        self.url = str(config.server_url)
        self.hz = float(config.hz)
        if self.hz <= 0:
            raise ValueError(f"hz must be positive, got {self.hz}")

        self.action_scale = as_array(config.action_scale, "action_scale", ((3,),))
        self.target_pose = as_array(config.target_pose, "target_pose", ((7,),))
        self.reset_pose = as_array(config.reset_pose, "reset_pose", ((7,),))
        self.reward_threshold = as_array(config.reward_threshold, "reward_threshold", ((6,),))
        if np.any(self.action_scale < 0):
            raise ValueError("action_scale values must be non-negative")
        if np.any(self.reward_threshold <= 0):
            raise ValueError("reward_threshold values must all be positive")
        for name, pose in (("target_pose", self.target_pose), ("reset_pose", self.reset_pose)):
            quat_norm = np.linalg.norm(pose[3:])
            if not np.isclose(quat_norm, 1.0, atol=1e-3):
                raise ValueError(f"{name} quaternion must be normalized, norm={quat_norm}")

        self.camera_configs = as_dict(
            config.realsense_cameras,
            "realsense_cameras",
        )
        if not self.camera_configs:
            raise ValueError("realsense_cameras must contain at least one camera")
        for camera_name, camera_config in self.camera_configs.items():
            if not isinstance(camera_config, Mapping):
                raise TypeError(
                    f"Camera config {camera_name!r} must be a mapping, "
                    f"got {type(camera_config).__name__}"
                )
            if "serial_number" not in camera_config:
                raise ValueError(
                    f"Camera config {camera_name!r} requires serial_number"
                )

        crop_config = as_dict(config.image_crop, "image_crop")
        unknown_crop_keys = crop_config.keys() - self.camera_configs.keys()
        if unknown_crop_keys:
            raise ValueError(
                f"image_crop contains unknown cameras: {sorted(unknown_crop_keys)}"
            )
        self.image_crop: dict[str, tuple[int, int, int, int]] = {}
        for camera_name, bounds in crop_config.items():
            if len(bounds) != 4:
                raise ValueError(
                    f"image_crop.{camera_name} must be [top, bottom, left, right]"
                )
            top, bottom, left, right = (int(value) for value in bounds)
            if min(top, left) < 0 or bottom <= top or right <= left:
                raise ValueError(f"Invalid crop bounds for {camera_name!r}: {bounds}")
            self.image_crop[camera_name] = (top, bottom, left, right)

        self.compliance_param = as_dict(config.compliance_param, "compliance_param")
        self.precision_param = as_dict(config.precision_param, "precision_param")
        self.max_episode_length = int(config.max_episode_length)
        self.display_image = bool(config.display_image)

        xyz_limit_low = as_array(config.abs_xyz_limit_low, "abs_xyz_limit_low", ((3,),))
        xyz_limit_high = as_array(config.abs_xyz_limit_high, "abs_xyz_limit_high", ((3,),))
        self.xyz_bounding_box = gym.spaces.Box(
            self.target_pose[:3] - xyz_limit_low,
            self.target_pose[:3] + xyz_limit_high,
            dtype=np.float64,
        )
        if np.any(self.xyz_bounding_box.low >= self.xyz_bounding_box.high):
            raise ValueError("The configured xyz safety bounds must have positive width")

        self.lastsent = time.time()
        self.pose_clip = bool(config.pose_clip)
        self.random_reset = bool(config.random_reset)

        self.random_x_range = float(config.random_x_range)
        self.random_x_range_neg = float(config.random_x_range_neg)
        self.random_y_range = float(config.random_y_range)
        self.random_y_range_neg = float(config.random_y_range_neg)
        self.random_z_range = float(config.random_z_range)
        self.random_z_range_neg = float(config.random_z_range_neg)

        self.random_rx_range = float(config.random_rx_range)
        self.random_rx_range_neg = float(config.random_rx_range_neg)
        self.random_ry_range = float(config.random_ry_range)
        self.random_ry_range_neg = float(config.random_ry_range_neg)
        self.random_rz_range = float(config.random_rz_range)
        self.random_rz_range_neg = float(config.random_rz_range_neg)

        self.joint_reset_period = int(config.joint_reset_period)
        if self.joint_reset_period < 0:
            raise ValueError(
                f"joint_reset_period must be non-negative, "
                f"got {self.joint_reset_period}"
            )

        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((6,), dtype=np.float32) * -1,
            np.ones((6,), dtype=np.float32),
        )
        state_space_dict = {
            "tcp_pose": gym.spaces.Box(
                -np.inf, np.inf, shape=(7,)
            ),  # xyz + quat
            "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
            "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
        }
        
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_space_dict),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for key in self.camera_configs}
                ),
            }
        )
        self.cycle_count = 0
        self.fake_env = bool(fake_env)

        if self.fake_env:
            self.currpos = self.reset_pose.copy()
            self.currvel = np.zeros(6, dtype=np.float64)
            self.currforce = np.zeros(3, dtype=np.float64)
            self.currtorque = np.zeros(3, dtype=np.float64)
            self.curr_path_length = 0
            return

        self.cap = None
        self.init_cameras(self.camera_configs)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        self.robot = FrankaApiClient(self.url)

        self._update_currpos()
        print("Initialized Franka")

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.currpos[:3] + action[:3] * self.action_scale[0]
        self.nextpos[3:] = (Rotation.from_rotvec(action[3:6] * self.action_scale[1]) * Rotation.from_quat(self.currpos[3:])).as_quat()
        self.nextpos = self._clip_safety_box(self.nextpos)
        if self.fake_env:
            self.currpos = self.nextpos.copy()
            self.currvel = np.concatenate(
                [
                    action[:3] * self.action_scale[0] * self.hz,
                    action[3:6] * self.action_scale[1] * self.hz,
                ]
            )
        else:
            self._send_pos_command(self.nextpos)

        self.curr_path_length += 1
        if not self.fake_env:
            dt = time.time() - start_time
            time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        terminated = bool(reward)
        truncated = self.curr_path_length >= self.max_episode_length and not terminated
        return ob, int(reward), terminated, truncated, {"succeed": terminated}

    def compute_reward(self, obs) -> bool:
        tcp_pose = np.asarray(obs["state"]["tcp_pose"], dtype=np.float64)
        position_error = np.abs(tcp_pose[:3] - self.target_pose[:3])
        rotation_error = np.abs(
            (
                Rotation.from_quat(tcp_pose[3:])
                * Rotation.from_quat(self.target_pose[3:]).inv()
            ).as_rotvec()
        )
        error = np.concatenate([position_error, rotation_error])
        return bool(np.all(error < self.reward_threshold))

    def get_im(self) -> Dict[str, np.ndarray]:
        if self.fake_env:
            return {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space["images"].spaces.items()
            }

        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                if key in self.image_crop:
                    top, bottom, left, right = self.image_crop[key]
                    cropped_rgb = rgb[top:bottom, left:right]
                else:
                    cropped_rgb = rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.camera_configs)
                return self.get_im()

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        steps = int(timeout * self.hz)
        self._update_currpos()

        start_pos = self.currpos[:3]
        start_quat = self.currpos[3:]
        start_quat = start_quat / np.linalg.norm(start_quat)

        goal_pos = goal[:3]
        goal_quat = goal[3:]
        goal_quat = goal_quat / np.linalg.norm(goal_quat)

        if np.dot(start_quat, goal_quat) < 0:
            goal_quat = -goal_quat

        pos_path = np.linspace(start_pos, goal_pos, steps)
        rotations = Rotation.from_quat([start_quat, goal_quat])
        slerp = Slerp([0, 1], rotations)
        times = np.linspace(0, 1, steps)
        quat_path = slerp(times).as_quat()
        
        # Combine position and quaternion paths
        path = np.hstack([pos_path, quat_path])
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        if self.fake_env:
            self.currpos = self.reset_pose.copy()
            self.currvel = np.zeros(6, dtype=np.float64)
            return

        if joint_reset:
            self.robot.joint_reset()

        # Change to precision mode for reset    
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        self.robot.update_param(self.precision_param)
        time.sleep(0.5)

        # Perform Carteasian reset
        if self.random_reset:
            reset_pose = self.reset_pose.copy()
            t_delta = np.array(
                [
                    np.random.uniform(self.random_x_range_neg, self.random_x_range),
                    np.random.uniform(self.random_y_range_neg, self.random_y_range),
                    np.random.uniform(self.random_z_range_neg, self.random_z_range),
                ]
            )
            reset_pose[:3] += t_delta
            
            quat_reset = reset_pose[3:].copy()
            euler_delta = np.array(
                [
                    np.random.uniform(self.random_rx_range_neg, self.random_rx_range),
                    np.random.uniform(self.random_ry_range_neg, self.random_ry_range),
                    np.random.uniform(self.random_rz_range_neg, self.random_rz_range),
                ]
            )
            reset_pose[3:] = (
                Rotation.from_euler("xyz", euler_delta) * Rotation.from_quat(quat_reset)
            ).as_quat()

            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.reset_pose.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        self.robot.update_param(self.compliance_param)

    def reset(self, joint_reset=False, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        self.cycle_count += 1
        periodic_joint_reset = (
            self.joint_reset_period > 0
            and self.cycle_count % self.joint_reset_period == 0
        )
        if not self.fake_env:
            self.robot.update_param(self.compliance_param)
        self.go_to_reset(joint_reset=joint_reset or periodic_joint_reset)
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        return obs, {"succeed": False}

    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        pose = pose.copy()

        if self.pose_clip:
            # Translation clip
            pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
            # Rotation clip
            delta_R = Rotation.from_quat(pose[3:]) * Rotation.from_quat(self.reset_pose[3:]).inv()
            delta_euler = delta_R.as_euler("xyz")
            delta_euler = np.clip(
                delta_euler,
                [self.random_rx_range_neg, self.random_ry_range_neg, self.random_rz_range_neg],
                [self.random_rx_range, self.random_ry_range, self.random_rz_range],
            )
            pose[3:] = (
                Rotation.from_euler("xyz", delta_euler)
                * Rotation.from_quat(self.reset_pose[3:])
            ).as_quat()

        return pose

    def _recover(self):
        self.robot.clear_errors()

    def _send_pos_command(self, pos: np.ndarray):
        self.robot.servo_pose(pos)

    def _update_currpos(self):
        if self.fake_env:
            return
        state = self.robot.get_state()
        self.currpos = np.array(state["pose"])
        self.currvel = np.array(state["vel"])

        self.currforce = np.array(state["force"])
        self.currtorque = np.array(state["torque"])
        self.currjacobian = np.reshape(np.array(state["jacobian"]), (6, 7))

        self.q = np.array(state["q"])
        self.dq = np.array(state["dq"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if self.fake_env:
            return
        self.close_cameras()
        self.robot.close()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
