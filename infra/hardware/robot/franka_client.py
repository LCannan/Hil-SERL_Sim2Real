from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import requests


ArrayLike = Union[float, int, Sequence[float], np.ndarray]

@dataclass(frozen=True)
class FrankaClientConfig:
    base_url: str = "http://127.0.0.1:5000"
    timeout: float = 5.0


class FrankaApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 5.0):
        self._config = FrankaClientConfig(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    @staticmethod
    def _to_ndarray(payload: Any) -> np.ndarray:
        return np.asarray(payload, dtype=float)

    def _post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        url = f"{self._config.base_url}/{endpoint.lstrip('/')}"
        response = self._session.post(
            url,
            json=json_data,
            timeout=self._config.timeout if timeout is None else timeout,
        )
        response.raise_for_status()

        # Some endpoints only return a plain string, keep that behaviour.
        if not response.text:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def set_load(
        self,
        mass: float,
        center_of_mass: ArrayLike,
        load_inertia: ArrayLike,
    ) -> None:
        self._post(
            "/set_load",
            {
                "mass": float(mass),
                "center_of_mass": self._as_array(center_of_mass, (3,)).tolist(),
                "load_inertia": self._as_array(load_inertia, (9,)).tolist(),
            },
        )

    def start_impedance(self) -> None:
        self._post("/startimp")

    def stop_impedance(self) -> None:
        self._post("/stopimp")

    def getpos(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getpos")["pose"])

    def get_vel(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getvel")["vel"])

    def get_force(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getforce")["force"])

    def get_torque(self) -> np.ndarray:
        return self._to_ndarray(self._post("/gettorque")["torque"])

    def get_q(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getq")["q"])

    def get_dq(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getdq")["dq"])

    def get_jacobian(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getjacobian")["jacobian"])

    def get_pos_euler(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getpos_euler")["pose"])

    def get_state(self) -> Dict[str, np.ndarray]:
        data = self._post("/getstate")
        return {
            "pose": self._to_ndarray(data["pose"]),
            "vel": self._to_ndarray(data["vel"]),
            "force": self._to_ndarray(data["force"]),
            "torque": self._to_ndarray(data["torque"]),
            "q": self._to_ndarray(data["q"]),
            "dq": self._to_ndarray(data["dq"]),
            "jacobian": self._to_ndarray(data["jacobian"]),
        }

    def servo_pose(self, arr: ArrayLike) -> None:
        pose_arr = self._as_array(arr, (7,))
        self._post("/pose", {"arr": pose_arr.tolist()})

    def joint_reset(self) -> None:
        self._post("/jointreset", timeout=60.0)

    def clear_errors(self) -> None:
        self._post("/clearerr")

    def update_param(self, values: Mapping[str, Any]) -> None:
        payload: Dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.astype(float).tolist()
            elif isinstance(value, (list, tuple)):
                payload[key] = [
                    float(item) if isinstance(item, (float, int, np.floating, np.integer)) else item
                    for item in value
                ]
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                payload[key] = float(value)
            else:
                payload[key] = value
        self._post("/update_param", payload)

    def close(self) -> None:
        self._session.close()

    def _as_array(self, values: ArrayLike, 
                  shape: Optional[tuple] = None) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if shape is not None and arr.shape != shape:
            raise ValueError(f"Expected shape {shape}, got {arr.shape}")
        return arr
