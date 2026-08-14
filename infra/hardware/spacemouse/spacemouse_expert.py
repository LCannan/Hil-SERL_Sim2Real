import multiprocessing
import numpy as np
from infra.hardware.spacemouse import pyspacemouse
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()
        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]
        # Cumulative 0->1 transitions per button.  Counted here rather than in
        # the consumer because this loop free-runs against the HID reports while
        # a control step is 50 ms at 20 Hz: a crisp click can begin and end
        # between two consecutive samples of the button *level* and be missed
        # entirely.  A click that toggles state cannot afford to be missed --
        # unlike the old hold-to-close binding, a dropped edge leaves the
        # operator's next click inverted rather than merely late.
        self.latest_data["press_counts"] = [0, 0, 0, 0]

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        press_counts = [0, 0, 0, 0]
        # Tracked locally and updated *only* from a genuine read, so the
        # zero-filled buttons written by the exception handlers below cannot
        # synthesize a release/press pair and toggle the gripper on their own.
        previous = [0, 0, 0, 0]
        while True:
            try:
                state = pyspacemouse.read_all()
                action = [0.0] * 6
                buttons = [0, 0, 0, 0]

                if len(state) == 2:
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw,
                        -state[1].y, state[1].x, state[1].z,
                        -state[1].roll, -state[1].pitch, -state[1].yaw
                    ]
                    buttons = state[0].buttons + state[1].buttons
                elif len(state) == 1:
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw
                    ]
                    buttons = state[0].buttons

                for index, level in enumerate(buttons[: len(press_counts)]):
                    if level and not previous[index]:
                        press_counts[index] += 1
                    previous[index] = int(bool(level))

                try:
                    self.latest_data["action"] = action
                    self.latest_data["buttons"] = buttons
                    self.latest_data["press_counts"] = list(press_counts)
                except (BrokenPipeError, ConnectionError, OSError):
                    # Manager connection has been closed, exit gracefully
                    break

            except (BrokenPipeError, ConnectionError, OSError):
                # Manager connection has been closed, exit gracefully
                break
            except Exception as e:
                # If reading fails, continue with zero action
                # But check if we can still write to the manager
                try:
                    self.latest_data["action"] = [0.0] * 6
                    self.latest_data["buttons"] = [0, 0, 0, 0]
                except (BrokenPipeError, ConnectionError, OSError):
                    # Manager connection lost, exit
                    break

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        # manager.dict() 已经是进程安全的，不需要锁
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def get_press_counts(self) -> list:
        """Cumulative per-button press counts, for edge-triggered bindings.

        Separate from `get_action` so the two legacy wrappers in
        `infra/wrappers/intervention.py`, which read that method directly, keep
        their existing return arity.
        """
        return list(self.latest_data["press_counts"])
    
    def close(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        if hasattr(self, 'manager'):
            try:
                self.manager.shutdown()
            except:
                pass
