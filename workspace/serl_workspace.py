from omegaconf import OmegaConf

from algorithm.wrappers.chunking import ChunkingWrapper
from algorithm.wrappers.serl_obs import SERLObsWrapper
from infra.wrappers.relative_frame import RelativeFrame
from infra.wrappers.robot_pose import Quat2RotvecWrapper
from .base_workspace import BaseWorkspace


class SERLWorkspace(BaseWorkspace):
    _JOB_NAME = "serl"

    def __init__(self, task_name, overrides=None):
        super().__init__(task_name, overrides)

    def get_environment(
        self,
        fake_env: bool = False,
        seed: int | None = None,
    ):
        env = super().get_environment(fake_env=fake_env, seed=seed)

        wrappers = self._config.wrappers
        # `.get` rather than attribute access: the four pre-existing task
        # configs have no `hil` block, and every `wrappers.*` key below is read
        # unconditionally, so a new unconditional key would break all of them.
        hil = self._config.get("hil")
        if hil is not None and hil.get("enabled") and not fake_env:
            from infra.experts import make_expert
            from infra.wrappers.intervention import ExpertIntervention

            expert_kwargs = hil.get("expert_kwargs") or {}
            if OmegaConf.is_config(expert_kwargs):
                expert_kwargs = OmegaConf.to_container(expert_kwargs, resolve=True)

            # One TeleopDisplay serves both the HUD and the keyboard expert's
            # cv2 backend: a cv2 window only reports keys to whoever calls
            # waitKey on the thread that owns it, so a second window would
            # leave one of the two permanently starved of input.
            display = self._make_teleop_display(hil)
            if display is not None:
                expert_kwargs.setdefault("display", display)

            # The expert is mounted innermost and receives the raw environment:
            # the scripted experts read privileged simulator state, and both
            # observe and act in the base frame before RelativeFrame rotates
            # things into the policy's frame.
            expert = make_expert(
                str(hil.expert),
                env=env,
                action_dim=int(env.action_space.shape[0]),
                seed=int(seed) if seed is not None else 0,
                **expert_kwargs,
            )
            env = ExpertIntervention(
                env,
                expert=expert,
                trigger=str(hil.get("trigger", "disagreement")),
                disagreement_threshold=float(hil.get("disagreement_threshold", 0.6)),
                min_takeover_steps=int(hil.get("min_takeover_steps", 5)),
                max_intervention_ratio=float(hil.get("max_intervention_ratio", 0.4)),
                intervention_decay_steps=int(hil.get("intervention_decay_steps", 0)),
                manual_deadband=float(hil.get("manual_deadband", 1e-3)),
            )
            if display is not None:
                from infra.wrappers.teleop_hud import TeleopHUD

                # Outside ExpertIntervention but inside RelativeFrame: see the
                # frame and observation-shape reasons in that module.
                env = TeleopHUD(
                    env,
                    display=display,
                    target_pose=self._config.environment.get("config", {}).get(
                        "target_pose"
                    ),
                )
        elif wrappers.spacemouse and not fake_env:
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

    @staticmethod
    def _make_teleop_display(hil):
        """Build the teleop window, or None when it cannot or should not open.

        Returning None on a headless host is deliberate rather than letting
        cv2 fail later: with no display, Qt aborts the whole process inside
        `cv2.imshow` before any Python exception exists to catch.
        """
        if not hil.get("hud"):
            return None

        from infra.experts.keyboard import KEY_LEGEND
        from infra.utils.teleop_display import DISPLAY_AVAILABLE, TeleopDisplay

        if not DISPLAY_AVAILABLE:
            print(
                "hil.hud is set but no DISPLAY/WAYLAND_DISPLAY was found; "
                "running without the teleop window."
            )
            return None
        return TeleopDisplay(
            enabled=True,
            scale=int(hil.get("hud_scale", 3)),
            legend=KEY_LEGEND,
        )
