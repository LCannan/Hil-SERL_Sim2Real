import logging
import os
import signal
import subprocess
import threading
import time

from absl import app, flags
from flask import Flask, jsonify, request
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from action_msgs.msg import GoalStatus
from franka_msgs.action import ErrorRecovery
from franka_msgs.msg import FrankaRobotState
from franka_msgs.srv import SetLoad
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float64MultiArray


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "robot_ip",
    "172.16.0.2",
    "IP address of the Franka controller box.",
)
flags.DEFINE_string(
    "robot_type",
    "fr3",
    "Franka robot type.",
)
flags.DEFINE_string(
    "namespace",
    "",
    "Optional ROS 2 namespace.",
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, 0, 2, 0],
    "Must match target_joint_positions in the controller YAML.",
)
flags.DEFINE_string(
    "flask_url",
    "127.0.0.1",
    "Address on which Flask listens.",
)
flags.DEFINE_integer(
    "flask_port",
    5000,
    "Port on which Flask listens.",
)


_LAUNCH_STOP_STEPS = (
    (signal.SIGINT, 5.0),
    (signal.SIGTERM, 3.0),
    (signal.SIGKILL, 5.0),
)
_shutdown_started = threading.Event()


def _handle_shutdown_signal(_signum, _frame):
    if _shutdown_started.is_set():
        return

    _shutdown_started.set()
    raise KeyboardInterrupt


def ros_name(namespace, name):
    """Construct an absolute ROS 2 interface name."""
    namespace = namespace.strip().strip("/")
    name = name.strip("/")

    if namespace:
        return f"/{namespace}/{name}"

    return f"/{name}"


class FrankaServer(Node):
    def __init__(
        self,
        robot_ip,
        robot_type,
        namespace,
        reset_joint_target,
    ):
        node_namespace = namespace.strip().strip("/") or None

        super().__init__(
            "franka_control_api",
            namespace=node_namespace,
        )

        self.robot_ip = robot_ip
        self.robot_type = robot_type
        self.namespace_name = namespace.strip().strip("/")
        self.reset_joint_target = np.asarray(
            reset_joint_target,
            dtype=float,
        )

        if self.reset_joint_target.shape != (7,):
            raise ValueError(
                "reset_joint_target must contain seven values."
            )

        if not np.all(np.isfinite(self.reset_joint_target)):
            raise ValueError(
                "reset_joint_target must contain finite values."
            )

        self.state_lock = threading.Lock()
        self.state_sequence = 0

        self.pos = np.zeros(7)
        self.pos[6] = 1.0
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.jacobian = np.zeros((6, 7))
        self.vel = np.zeros(6)

        self.impedance_process = None
        self.joint_process = None
        self.process_lock = threading.RLock()
        self.shutting_down = False

        self.eepub = self.create_publisher(
            PoseStamped,
            ros_name(
                namespace,
                "cartesian_impedance_controller/equilibrium_pose",
            ),
            10,
        )

        self.state_sub = self.create_subscription(
            FrankaRobotState,
            ros_name(
                namespace,
                "franka_robot_state_broadcaster/robot_state",
            ),
            self._set_currpos,
            qos_profile_sensor_data,
        )

        self.jacobian_sub = self.create_subscription(
            Float64MultiArray,
            ros_name(
                namespace,
                "cartesian_impedance_controller/franka_jacobian",
            ),
            self._set_jacobian,
            qos_profile_sensor_data,
        )

        self.error_recovery_client = ActionClient(
            self,
            ErrorRecovery,
            ros_name(
                namespace,
                "action_server/error_recovery",
            ),
        )

        self.set_load_client = self.create_client(
            SetLoad,
            ros_name(
                namespace,
                "service_server/set_load",
            ),
        )

        self.parameter_client = self.create_client(
            SetParameters,
            ros_name(
                namespace,
                "cartesian_impedance_controller/set_parameters",
            ),
        )

    @staticmethod
    def _wait_future(
        future,
        timeout,
        description,
    ):
        """Wait for a future while the executor spins in another thread."""
        finished = threading.Event()
        future.add_done_callback(lambda _: finished.set())

        if not finished.wait(timeout):
            raise TimeoutError(
                f"Timed out waiting for {description}."
            )

        exception = future.exception()
        if exception is not None:
            raise RuntimeError(
                f"{description} failed: {exception}"
            )

        return future.result()

    def _set_currpos(self, msg):
        joint_state = msg.measured_joint_state

        if (
            len(joint_state.position) < 7
            or len(joint_state.velocity) < 7
        ):
            self.get_logger().warning(
                "Received incomplete Franka joint state."
            )
            return

        pose = msg.o_t_ee.pose
        wrench = msg.k_f_ext_hat_k.wrench

        with self.state_lock:
            self.pos = np.array(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=float,
            )

            self.q = np.asarray(
                joint_state.position[:7],
                dtype=float,
            )
            self.dq = np.asarray(
                joint_state.velocity[:7],
                dtype=float,
            )

            self.force = np.array(
                [
                    wrench.force.x,
                    wrench.force.y,
                    wrench.force.z,
                ],
                dtype=float,
            )
            self.torque = np.array(
                [
                    wrench.torque.x,
                    wrench.torque.y,
                    wrench.torque.z,
                ],
                dtype=float,
            )

            self.vel = self.jacobian @ self.dq
            self.state_sequence += 1

    def _set_jacobian(self, msg):
        if len(msg.data) != 42:
            self.get_logger().warning(
                f"Expected 42 Jacobian values, got {len(msg.data)}."
            )
            return
        jacobian = np.asarray(
            msg.data,
            dtype=float,
        ).reshape(
            (6, 7),
            order="F",
        )

        with self.state_lock:
            self.jacobian = jacobian
            self.vel = self.jacobian @ self.dq

    def get_state(self):
        with self.state_lock:
            return {
                "pose": self.pos.copy(),
                "vel": self.vel.copy(),
                "force": self.force.copy(),
                "torque": self.torque.copy(),
                "q": self.q.copy(),
                "dq": self.dq.copy(),
                "jacobian": self.jacobian.copy(),
                "sequence": self.state_sequence,
            }

    def move(self, pose):
        pose = np.asarray(
            pose,
            dtype=float,
        )

        if pose.shape != (7,):
            raise ValueError(
                "Pose must be [x, y, z, qx, qy, qz, qw]."
            )

        if not np.all(np.isfinite(pose)):
            raise ValueError(
                "Pose must contain finite values."
            )

        if np.linalg.norm(pose[3:]) < 1e-8:
            raise ValueError(
                "Pose quaternion must be non-zero."
            )

        msg = PoseStamped()
        msg.header.frame_id = "base"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = float(pose[2])

        msg.pose.orientation.x = float(pose[3])
        msg.pose.orientation.y = float(pose[4])
        msg.pose.orientation.z = float(pose[5])
        msg.pose.orientation.w = float(pose[6])

        self.eepub.publish(msg)

    def clear(self, timeout=10.0):
        """Run the Franka ROS 2 error recovery action."""
        if not self.error_recovery_client.wait_for_server(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                "Error recovery action is not available."
            )

        goal_future = self.error_recovery_client.send_goal_async(
            ErrorRecovery.Goal()
        )

        goal_handle = self._wait_future(
            goal_future,
            timeout,
            "error recovery goal",
        )

        if not goal_handle.accepted:
            raise RuntimeError(
                "Error recovery goal was rejected."
            )

        result = self._wait_future(
            goal_handle.get_result_async(),
            timeout,
            "error recovery result",
        )

        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                f"Error recovery failed with status {result.status}."
            )

    def set_load(
        self,
        mass,
        center_of_mass,
        load_inertia,
        timeout=10.0,
    ):
        center_of_mass = np.asarray(
            center_of_mass,
            dtype=float,
        )
        load_inertia = np.asarray(
            load_inertia,
            dtype=float,
        )

        if center_of_mass.shape != (3,):
            raise ValueError(
                "center_of_mass must contain three values."
            )

        if load_inertia.shape != (9,):
            raise ValueError(
                "load_inertia must contain nine values."
            )

        if not np.all(np.isfinite(center_of_mass)):
            raise ValueError(
                "center_of_mass must contain finite values."
            )

        if not np.all(np.isfinite(load_inertia)):
            raise ValueError(
                "load_inertia must contain finite values."
            )

        if not self.set_load_client.wait_for_service(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                "set_load service is not available."
            )

        req = SetLoad.Request()
        req.mass = float(mass)
        req.center_of_mass = center_of_mass.tolist()
        req.load_inertia = load_inertia.tolist()

        response = self._wait_future(
            self.set_load_client.call_async(req),
            timeout,
            "set_load response",
        )

        if not response.success:
            raise RuntimeError(
                response.error or "Failed to set load."
            )

    def update_parameters(
        self,
        values,
        timeout=10.0,
    ):
        if not self.parameter_client.wait_for_service(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                "Controller parameter service is not available."
            )

        parameters = []

        for name, value in values.items():
            if isinstance(value, int) and not isinstance(value, bool):
                value = float(value)

            elif isinstance(value, list):
                value = [
                    float(item)
                    if isinstance(item, (int, float))
                    else item
                    for item in value
                ]

            parameters.append(
                Parameter(
                    name=name,
                    value=value,
                ).to_parameter_msg()
            )

        req = SetParameters.Request()
        req.parameters = parameters

        response = self._wait_future(
            self.parameter_client.call_async(req),
            timeout,
            "controller parameter response",
        )

        errors = [
            result.reason
            for result in response.results
            if not result.successful
        ]

        if errors:
            raise RuntimeError("; ".join(errors))

    def _launch_arguments(self):
        launch_arguments = [
            f"robot_ip:={self.robot_ip}",
            f"robot_type:={self.robot_type}",
            "load_gripper:=false",
        ]

        if self.namespace_name:
            launch_arguments.append(
                f"namespace:={self.namespace_name}"
            )

        return launch_arguments

    @staticmethod
    def _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

        # killpg() also succeeds while a process group contains only zombies.
        # Ignore those entries so completed ROS launch trees are not reported
        # as live while their final status is being reaped.
        try:
            process_entries = os.scandir("/proc")
        except OSError:
            return True

        with process_entries:
            for entry in process_entries:
                if not entry.name.isdigit():
                    continue

                try:
                    with open(
                        f"/proc/{entry.name}/stat",
                        encoding="utf-8",
                    ) as stat_file:
                        process_stat = stat_file.read()

                    command_end = process_stat.rfind(")")
                    stat_fields = process_stat[
                        command_end + 2:
                    ].split()
                    process_state = stat_fields[0]
                    process_group = int(stat_fields[2])
                except (
                    OSError,
                    IndexError,
                    UnicodeError,
                    ValueError,
                ):
                    continue

                if (
                    process_group == process_group_id
                    and process_state not in {"Z", "X"}
                ):
                    return True

        return False

    def _log_process_message(self, level, message):
        if rclpy.ok():
            getattr(self.get_logger(), level)(message)
            return

        getattr(logging.getLogger(__name__), level)(message)

    @classmethod
    def _wait_for_process_group(cls, process, timeout):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            process.poll()

            if not cls._process_group_exists(process.pid):
                return True

            time.sleep(0.1)

        process.poll()
        return not cls._process_group_exists(process.pid)

    def _start_launch(self, launch_file):
        return subprocess.Popen(
            [
                "ros2",
                "launch",
                "serl_franka_controllers_ros2",
                launch_file,
                *self._launch_arguments(),
            ],
            start_new_session=True,
        )

    def _stop_launch(self, process, description):
        if process is None:
            return

        process_group_id = process.pid

        for stop_signal, timeout in _LAUNCH_STOP_STEPS:
            process.poll()

            if not self._process_group_exists(process_group_id):
                break

            self._log_process_message(
                "info",
                f"Stopping {description} process group "
                f"{process_group_id} with {stop_signal.name}.",
            )

            try:
                os.killpg(process_group_id, stop_signal)
            except ProcessLookupError:
                break
            except PermissionError as error:
                self._log_process_message(
                    "error",
                    f"Cannot stop {description} process group "
                    f"{process_group_id}: {error}",
                )
                return

            if self._wait_for_process_group(process, timeout):
                break

        process.poll()

        if self._process_group_exists(process_group_id):
            self._log_process_message(
                "error",
                f"{description} process group {process_group_id} "
                "still has live members after SIGKILL.",
            )

    def start_impedance(self):
        """Launch impedance.launch.py."""
        with self.process_lock:
            if self.shutting_down:
                return

            if self.impedance_process is not None:
                if (
                    self.impedance_process.poll() is None
                    and self._process_group_exists(
                        self.impedance_process.pid
                    )
                ):
                    return

                self._stop_launch(
                    self.impedance_process,
                    "impedance launch",
                )

            self.impedance_process = self._start_launch(
                "impedance.launch.py"
            )
            time.sleep(5.0)

    def stop_impedance(self):
        with self.process_lock:
            self._stop_launch(
                self.impedance_process,
                "impedance launch",
            )
            self.impedance_process = None

    def start_joint_controller(self):
        """Launch joint.launch.py."""
        with self.process_lock:
            if self.shutting_down:
                return

            if self.joint_process is not None:
                if (
                    self.joint_process.poll() is None
                    and self._process_group_exists(
                        self.joint_process.pid
                    )
                ):
                    return

                self._stop_launch(
                    self.joint_process,
                    "joint launch",
                )

            self.joint_process = self._start_launch(
                "joint.launch.py"
            )
            time.sleep(5.0)

    def stop_joint_controller(self):
        with self.process_lock:
            self._stop_launch(
                self.joint_process,
                "joint launch",
            )
            self.joint_process = None

    def reset_joint(self):
        self.stop_impedance()
        self.start_joint_controller()

        initial_sequence = self.get_state()["sequence"]

        try:
            try:
                self.clear()
            except RuntimeError as error:
                self.get_logger().warning(
                    str(error)
                )

            deadline = time.monotonic() + 30.0

            while time.monotonic() < deadline:
                state = self.get_state()

                received_new_state = (
                    state["sequence"] > initial_sequence
                )
                reached_target = np.allclose(
                    state["q"],
                    self.reset_joint_target,
                    atol=1e-2,
                    rtol=1e-2,
                )

                if received_new_state and reached_target:
                    return

                time.sleep(0.2)

            raise TimeoutError(
                "Joint reset did not reach the configured target."
            )

        finally:
            self.stop_joint_controller()
            self.start_impedance()

    def shutdown(self):
        with self.process_lock:
            self.shutting_down = True

        self.stop_joint_controller()
        self.stop_impedance()


def main(_):
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGHUP, _handle_shutdown_signal)

    rclpy.init(
        args=[],
        signal_handler_options=SignalHandlerOptions.NO,
    )

    robot_server = FrankaServer(
        robot_ip=FLAGS.robot_ip,
        robot_type=FLAGS.robot_type,
        namespace=FLAGS.namespace,
        reset_joint_target=[
            float(value)
            for value in FLAGS.reset_joint_target
        ],
    )

    executor = MultiThreadedExecutor(
        num_threads=2
    )
    executor.add_node(robot_server)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )
    spin_thread.start()

    webapp = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @webapp.errorhandler(Exception)
    def handle_error(error):
        robot_server.get_logger().error(
            str(error)
        )
        return jsonify({"error": str(error)}), 500

    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        data = request.get_json()

        center_of_mass = data.get(
            "center_of_mass",
            data.get("F_x_center_load"),
        )

        robot_server.set_load(
            data["mass"],
            center_of_mass,
            data["load_inertia"],
        )

        return "Set Load"

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        robot_server.start_impedance()
        return "Started impedance"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot_server.stop_impedance()
        return "Stopped impedance"

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        pose = robot_server.get_state()["pose"]
        xyz = pose[:3]

        euler = R.from_quat(
            pose[3:]
        ).as_euler("xyz")

        return jsonify(
            {
                "pose": np.concatenate(
                    [xyz, euler]
                ).tolist()
            }
        )

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        state = robot_server.get_state()
        return jsonify(
            {"pose": state["pose"].tolist()}
        )

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        state = robot_server.get_state()
        return jsonify(
            {"vel": state["vel"].tolist()}
        )

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        state = robot_server.get_state()
        return jsonify(
            {"force": state["force"].tolist()}
        )

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        state = robot_server.get_state()
        return jsonify(
            {"torque": state["torque"].tolist()}
        )

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        state = robot_server.get_state()
        return jsonify(
            {"q": state["q"].tolist()}
        )

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        state = robot_server.get_state()
        return jsonify(
            {"dq": state["dq"].tolist()}
        )

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        state = robot_server.get_state()
        return jsonify(
            {
                "jacobian": state[
                    "jacobian"
                ].tolist()
            }
        )

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.reset_joint()
        return "Reset Joint"

    @webapp.route("/clearerr", methods=["POST"])
    def clear_errors():
        robot_server.clear()
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        robot_server.move(
            request.get_json()["arr"]
        )
        return "Moved"

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        state = robot_server.get_state()

        return jsonify(
            {
                "pose": state["pose"].tolist(),
                "vel": state["vel"].tolist(),
                "force": state["force"].tolist(),
                "torque": state["torque"].tolist(),
                "q": state["q"].tolist(),
                "dq": state["dq"].tolist(),
                "jacobian": state[
                    "jacobian"
                ].tolist(),
            }
        )

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        robot_server.update_parameters(
            request.get_json()
        )
        return "Updated compliance parameters"

    try:
        robot_server.start_impedance()

        try:
            robot_server.update_parameters(
                {"publish_jacobian": True}
            )
        except RuntimeError as error:
            robot_server.get_logger().warning(
                str(error)
            )

        webapp.run(
            host=FLAGS.flask_url,
            port=FLAGS.flask_port,
            threaded=True,
            use_reloader=False,
        )

    finally:
        _shutdown_started.set()
        robot_server.shutdown()
        executor.shutdown()
        robot_server.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    app.run(main)
