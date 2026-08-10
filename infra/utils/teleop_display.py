"""Single-threaded cv2 teleoperation window: HUD *and* keyboard source.

The two roles live in one class on purpose.  A cv2 window only reports keys to
whoever calls ``waitKey`` on the thread that created it, so a HUD rendering in
one place and a keyboard backend polling in another would fight over the same
window -- one of them would silently never see a keypress.  ``render`` performs
the single ``imshow``/``waitKey`` pair and stashes the key for :meth:`poll_key`.

Headless safety is why ``DISPLAY_AVAILABLE`` is computed from the environment
rather than by trying cv2 and catching the failure: with no display, Qt aborts
the *process* inside ``cv2.imshow`` (``qt.qpa.xcb: could not connect to
display``) before any Python exception can be raised, so ``try``/``except``
around it never runs.  Everything here degrades to a no-op instead.
"""

import os
from typing import Dict, Optional

import cv2
import numpy as np

# Probed once at import: the only reliable guard, see the module docstring.
DISPLAY_AVAILABLE = bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUMAN_COLOR = (60, 60, 255)  # BGR: red
_POLICY_COLOR = (80, 220, 80)  # BGR: green
_TEXT_COLOR = (240, 240, 240)
_PANEL_COLOR = (30, 30, 30)


class TeleopDisplay:
    """cv2 window showing camera views plus an operator HUD.

    Also the key source for :class:`~infra.experts.keyboard.KeyboardExpert`'s
    cv2 backend -- see the module docstring for why they share one window.

    When no display is available (or ``enabled=False``) every method is a
    no-op and :meth:`poll_key` always returns -1, so the same code path runs
    unchanged on a headless training host.
    """

    def __init__(
        self,
        window_name: str = "hil_teleop",
        enabled: bool = True,
        scale: int = 3,
        legend: str = "",
    ):
        self.window_name = str(window_name)
        self.scale = max(1, int(scale))
        self.legend = str(legend)
        self.active = bool(enabled) and DISPLAY_AVAILABLE
        self._last_key = -1
        self._window_open = False

    @property
    def available(self) -> bool:
        return self.active

    def render(
        self,
        frames: Dict[str, np.ndarray],
        hud: Optional[Dict] = None,
    ) -> None:
        """Draw one frame.  Cheap no-op when the display is unavailable."""
        if not self.active:
            return

        row = self._camera_row(frames)
        if row is None:
            return
        canvas = self._with_hud(row, hud or {})

        cv2.imshow(self.window_name, canvas)
        # The single waitKey: pumps the GUI event loop and collects the key
        # that poll_key() hands to the keyboard backend.
        self._last_key = cv2.waitKey(1) & 0xFF
        self._window_open = True

    def poll_key(self) -> int:
        """Consume the most recent keypress, or -1 if there was none."""
        key, self._last_key = self._last_key, -1
        return -1 if key in (-1, 255) else key

    def close(self) -> None:
        if self._window_open:
            cv2.destroyWindow(self.window_name)
            self._window_open = False

    def _camera_row(self, frames: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Concatenate the camera images side by side, upscaled and labelled."""
        tiles = []
        for name, image in frames.items():
            if image is None:
                continue
            array = np.asarray(image)
            while array.ndim > 3:  # ChunkingWrapper-style leading axes
                array = array[-1]
            if array.ndim != 3 or array.shape[-1] < 3:
                continue
            tile = np.ascontiguousarray(array[..., :3], dtype=np.uint8)
            if self.scale > 1:
                tile = cv2.resize(
                    tile,
                    (tile.shape[1] * self.scale, tile.shape[0] * self.scale),
                    interpolation=cv2.INTER_NEAREST,
                )
            tile = tile[..., ::-1]  # cv2 windows want BGR
            tile = np.ascontiguousarray(tile)
            cv2.putText(tile, str(name), (6, 18), _FONT, 0.5, _TEXT_COLOR, 1,
                        cv2.LINE_AA)
            tiles.append(tile)

        if not tiles:
            return None
        height = min(tile.shape[0] for tile in tiles)
        return np.concatenate([tile[:height] for tile in tiles], axis=1)

    def _with_hud(self, row: np.ndarray, hud: Dict) -> np.ndarray:
        """Stack a text panel under the camera row."""
        lines = self._hud_lines(hud)
        panel_height = 22 * len(lines) + 16
        panel = np.full((panel_height, row.shape[1], 3), _PANEL_COLOR, np.uint8)

        intervening = bool(hud.get("intervening"))
        for index, (text, highlight) in enumerate(lines):
            color = (
                (_HUMAN_COLOR if intervening else _POLICY_COLOR)
                if highlight
                else _TEXT_COLOR
            )
            cv2.putText(panel, text, (8, 24 + index * 22), _FONT, 0.5, color, 1,
                        cv2.LINE_AA)
        return np.concatenate([row, panel], axis=0)

    def _hud_lines(self, hud: Dict):
        """Build (text, is_control_line) pairs.  Missing fields are skipped."""
        intervening = bool(hud.get("intervening"))
        lines = [(f"CONTROL: {'HUMAN' if intervening else 'POLICY'}", True)]

        action = hud.get("action")
        if action is not None:
            lines.append((f"action  {self._fmt_vec(action)}", False))

        error = hud.get("pose_error")
        if error is not None and len(error) >= 6:
            pos_mm = np.asarray(error[:3], dtype=np.float64) * 1000.0
            rot = np.asarray(error[3:6], dtype=np.float64)
            lines.append((
                "pos err mm [{:6.1f}{:7.1f}{:7.1f}]  rot err [{:5.2f}{:6.2f}"
                "{:6.2f}]".format(*pos_mm, *rot),
                False,
            ))

        status = []
        for key, label, fmt in (
            ("step", "step", "{}"),
            ("episode", "ep", "{}"),
            ("episode_return", "return", "{:.2f}"),
        ):
            if hud.get(key) is not None:
                status.append(f"{label} {fmt.format(hud[key])}")
        if hud.get("succeed"):
            status.append("SUCCESS")
        if status:
            lines.append(("  ".join(status), False))

        if self.legend:
            lines.append((self.legend, False))
        return lines

    @staticmethod
    def _fmt_vec(vector) -> str:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        return "[" + " ".join(f"{value:6.2f}" for value in values[:6]) + "]"
