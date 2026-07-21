from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class BaseWorkspace:
    _CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
    _TASK_CONFIG_DIR = _CONFIG_DIR / "task"
    _JOB_NAME = "workspace"

    def __init__(
        self,
        task_name: str,
        overrides: Sequence[str] | None = None,
    ):
        self._config = self._load_config(task_name, overrides)

    @classmethod
    def available_tasks(cls) -> tuple[str, ...]:
        if not cls._TASK_CONFIG_DIR.is_dir():
            raise FileNotFoundError(
                f"Hydra task config directory not found: {cls._TASK_CONFIG_DIR}"
            )
        return tuple(sorted(path.stem for path in cls._TASK_CONFIG_DIR.glob("*.yaml")))

    @staticmethod
    def _task_override(override: str) -> str:
        override = override.strip()
        if not override:
            raise ValueError("Hydra overrides must not be empty")

        operator = ""
        while override[:1] in {"+", "~"}:
            operator += override[0]
            override = override[1:]

        if override.startswith(("task.", "hydra.")):
            return operator + override
        return f"{operator}task.{override}"

    @classmethod
    def _load_config(
        cls,
        task_name: str,
        overrides: Sequence[str] | None,
    ) -> DictConfig:
        tasks = cls.available_tasks()
        if task_name not in tasks:
            choices = ", ".join(tasks) or "<none>"
            raise ValueError(f"Unknown task {task_name!r}. Available tasks: {choices}")

        hydra_overrides = [f"task={task_name}"]
        hydra_overrides.extend(cls._task_override(item) for item in overrides or ())

        with initialize_config_dir(
            version_base=None,
            config_dir=str(cls._CONFIG_DIR),
            job_name=cls._JOB_NAME,
        ):
            root_config = compose(config_name="config", overrides=hydra_overrides)

        return OmegaConf.create(
            OmegaConf.to_container(root_config.task, resolve=True)
        )

    @property
    def name(self) -> str:
        return str(self._config.name)

    @property
    def raw_config(self) -> DictConfig:
        return self._config

    def __getattr__(self, name: str) -> Any:
        training = self._config.training
        if name not in training:
            raise AttributeError(
                f"{type(self).__name__} {self.name!r} has no attribute {name!r}"
            )

        value = training[name]
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def to_dict(self) -> dict[str, Any]:
        return OmegaConf.to_container(self._config, resolve=True)

    def get_environment(self, fake_env: bool = False, seed: int | None = None):
        kwargs = {"fake_env": fake_env}
        if seed is not None and "seed" in self._config.environment:
            kwargs["seed"] = seed
        return instantiate(self._config.environment, **kwargs)

    def get_agent(
        self,
        seed: int,
        sample_obs: Any,
        sample_action: Any,
    ):
        return instantiate(
            self._config.agent,
            seed=seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            _convert_="all",
        )
