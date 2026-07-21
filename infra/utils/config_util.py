from collections.abc import Mapping
from typing import Any

import numpy as np
from omegaconf import OmegaConf


def as_array(
    value: Any,
    name: str,
    valid_shapes: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape not in valid_shapes:
        raise ValueError(f"{name} must have shape {valid_shapes}, got {array.shape}")
    return array


def as_dict(value: Any, name: str) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)