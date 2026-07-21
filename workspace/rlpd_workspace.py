from algorithm.wrappers.chunking import ChunkingWrapper
from algorithm.wrappers.serl_obs import SERLObsWrapper
from infra.wrappers.relative_frame import RelativeFrame
from infra.wrappers.robot_pose import Quat2RotvecWrapper
from .base_workspace import BaseWorkspace


class RLPDWorkspace(BaseWorkspace):
    _JOB_NAME = "rlpd"

    def __init__(self, task_name, overrides=None):
        super().__init__(task_name, overrides)
        positive_training_fields = (
            "batch_size",
            "cta_ratio",
            "max_steps",
            "replay_buffer_capacity",
            "training_starts",
            "steps_per_update",
            "actor_update_period",
            "log_period",
        )
        for field in positive_training_fields:
            if int(self._config.training[field]) <= 0:
                raise ValueError(f"training.{field} must be positive")
        if int(self._config.training.batch_size) < 2:
            raise ValueError("training.batch_size must be at least 2")
        if int(self._config.wrappers.obs_horizon) != 1:
            raise ValueError(
                "Only wrappers.obs_horizon=1 is supported by the current encoders"
            )
        if self._config.wrappers.act_exec_horizon is not None:
            raise ValueError(
                "wrappers.act_exec_horizon must be null for single-action SAC"
            )
        if not self._config.wrappers.serl_observation:
            raise ValueError("wrappers.serl_observation must be enabled for SAC")

    def get_environment(
        self,
        fake_env: bool = False,
        seed: int | None = None,
    ):
        env = super().get_environment(fake_env=fake_env, seed=seed)

        wrappers = self._config.wrappers
        if wrappers.spacemouse and not fake_env:
            from infra.wrappers.intervention import SpacemouseIntervention

            env = SpacemouseIntervention(env)
        if wrappers.relative_frame:
            env = RelativeFrame(
                env,
                include_relative_pose=bool(wrappers.include_relative_pose),
            )
        if wrappers.quat_to_rotvec:
            env = Quat2RotvecWrapper(env)
        if wrappers.serl_observation:
            env = SERLObsWrapper(
                env,
                proprio_keys=list(self._config.training.proprio_keys),
            )
        if wrappers.chunking:
            env = ChunkingWrapper(
                env,
                obs_horizon=int(wrappers.obs_horizon),
                act_exec_horizon=wrappers.act_exec_horizon,
            )
        missing_image_keys = set(self._config.training.image_keys) - set(
            env.observation_space.spaces
        )
        if missing_image_keys:
            env.close()
            raise ValueError(
                "training.image_keys are missing from the wrapped observation "
                f"space: {sorted(missing_image_keys)}"
            )
        return env
