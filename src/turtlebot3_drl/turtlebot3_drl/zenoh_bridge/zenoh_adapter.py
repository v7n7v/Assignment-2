"""
Zenoh-based DRL adapter for turtlebot3_drlnav.

Replaces all ROS 2 communication (topics and services) with pure Zenoh
pub/sub and request/reply, allowing the DRL agent container to run with
ZERO ROS dependencies. Communication with the ROS 2 simulation
environment is handled by zenoh-bridge-ros2dds on the simulation side.

Subscribes to:
    rt/scan       - LiDAR data   (sensor_msgs/msg/LaserScan, CDR)
    rt/odom       - Robot pose    (nav_msgs/msg/Odometry, CDR)
    rt/goal_pose  - Goal position (geometry_msgs/msg/Pose, CDR)

Publishes to:
    rt/cmd_vel            - Velocity commands  (geometry_msgs/msg/Twist, CDR)
    tb/drl/step_request   - Step sync request  (JSON)
    tb/drl/metrics        - Training metrics   (JSON)

Step synchronization:
    tb/drl/step_response  - Step sync response (JSON, subscribed)

Pattern follows turtlebot-maze/detector/object_detector.py and
turtlebot-maze/slam/slam_bridge.py.
"""

import json
import math
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import zenoh

from .cdr_types import LaserScan, Odometry, Pose, Twist, Vector3


# ===================================================================== #
#                              Constants                                 #
# ===================================================================== #

# Default number of LiDAR samples expected by the DRL model.
NUM_SCAN_SAMPLES = 40

# Environment geometry (mirrors common/settings.py defaults).
ARENA_LENGTH = 4.2          # metres
ARENA_WIDTH = 4.2           # metres
MAX_GOAL_DISTANCE = math.sqrt(ARENA_LENGTH ** 2 + ARENA_WIDTH ** 2)
LIDAR_DISTANCE_CAP = 3.5    # metres
SPEED_LINEAR_MAX = 0.22     # m/s
SPEED_ANGULAR_MAX = 2.0     # rad/s

# Zenoh key-expressions (rt/ prefix is added by zenoh-bridge-ros2dds).
KEY_SCAN = "rt/scan"
KEY_ODOM = "rt/odom"
KEY_GOAL_POSE = "rt/goal_pose"
KEY_CMD_VEL = "rt/cmd_vel"
KEY_STEP_REQUEST = "tb/drl/step_request"
KEY_STEP_RESPONSE = "tb/drl/step_response"
KEY_METRICS = "tb/drl/metrics"

# Step synchronization timeout (seconds).
DEFAULT_STEP_TIMEOUT = 10.0


# ===================================================================== #
#                           Helper Functions                             #
# ===================================================================== #

def euler_from_quaternion(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Convert a quaternion to Euler angles (roll, pitch, yaw).

    Matches the implementation in common/utilities.py so heading values are
    identical to the ROS-based environment node.
    """
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


def _downsample_scan(ranges: List[float], target_count: int) -> List[float]:
    """Downsample a laser scan to *target_count* evenly-spaced samples.

    If the incoming scan already has *target_count* samples it is returned
    unchanged. Otherwise, we pick evenly spaced indices via linspace.
    """
    n = len(ranges)
    if n == target_count:
        return list(ranges)
    if n == 0:
        return [LIDAR_DISTANCE_CAP] * target_count
    indices = np.round(np.linspace(0, n - 1, target_count)).astype(int)
    return [ranges[i] for i in indices]


# ===================================================================== #
#                          ZenohDRLAdapter                               #
# ===================================================================== #

class ZenohDRLAdapter:
    """Bridge between the DRL agent and the ROS 2 simulation via Zenoh.

    All sensor data is cached in a thread-safe manner. The :meth:`step`
    method implements synchronous request/reply with the environment node
    running on the ROS side.

    Parameters
    ----------
    connect : str
        Zenoh router endpoint (e.g. ``"tcp/192.168.1.10:7447"``).
        Leave empty for multicast scouting.
    num_scan_samples : int
        Number of LiDAR samples the DRL model expects (default 40).
    step_timeout : float
        Maximum seconds to wait for a step response.
    enable_backward : bool
        Whether backward linear velocity is enabled.
    """

    def __init__(
        self,
        connect: str = "",
        num_scan_samples: int = NUM_SCAN_SAMPLES,
        step_timeout: float = DEFAULT_STEP_TIMEOUT,
        enable_backward: bool = False,
    ) -> None:
        self.num_scan_samples = num_scan_samples
        self.step_timeout = step_timeout
        self.enable_backward = enable_backward

        # ----- Cached sensor state (protected by _lock) ----- #
        self._lock = threading.Lock()
        self._scan_ranges: List[float] = [LIDAR_DISTANCE_CAP] * num_scan_samples
        self._obstacle_distance: float = LIDAR_DISTANCE_CAP
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_heading: float = 0.0
        self._goal_x: float = 0.0
        self._goal_y: float = 0.0
        self._goal_distance: float = MAX_GOAL_DISTANCE
        self._goal_angle: float = 0.0
        self._new_goal: bool = False

        # ----- Step synchronization ----- #
        self._step_event = threading.Event()
        self._step_response: Optional[Dict] = None

        # ----- Open Zenoh session ----- #
        conf = zenoh.Config()
        if connect:
            conf.insert_json5("connect/endpoints", json.dumps([connect]))

        self._session = zenoh.open(conf)

        # Publishers
        self._pub_cmd_vel = self._session.declare_publisher(KEY_CMD_VEL)
        self._pub_step_request = self._session.declare_publisher(KEY_STEP_REQUEST)
        self._pub_metrics = self._session.declare_publisher(KEY_METRICS)

        # Subscribers
        self._sub_scan = self._session.declare_subscriber(KEY_SCAN, self._on_scan)
        self._sub_odom = self._session.declare_subscriber(KEY_ODOM, self._on_odom)
        self._sub_goal = self._session.declare_subscriber(KEY_GOAL_POSE, self._on_goal_pose)
        self._sub_step_response = self._session.declare_subscriber(
            KEY_STEP_RESPONSE, self._on_step_response
        )

        print(f"ZenohDRLAdapter ready (scan_samples={num_scan_samples})")
        print(f"  connect:        {'multicast' if not connect else connect}")
        print(f"  step_timeout:   {step_timeout}s")
        print(f"  enable_backward: {enable_backward}")

    # ----------------------------------------------------------------- #
    #                        Zenoh Callbacks                             #
    # ----------------------------------------------------------------- #

    def _on_scan(self, sample: zenoh.Sample) -> None:
        """Deserialize LaserScan CDR, downsample, normalize, and cache."""
        try:
            msg = LaserScan.deserialize(bytes(sample.payload))
        except Exception as exc:
            print(f"[scan] CDR deserialize error: {exc}")
            return

        raw_ranges = list(msg.ranges)
        downsampled = _downsample_scan(raw_ranges, self.num_scan_samples)

        obstacle_dist = LIDAR_DISTANCE_CAP
        normalized: List[float] = []
        for r in downsampled:
            v = float(np.clip(r / LIDAR_DISTANCE_CAP, 0.0, 1.0))
            normalized.append(v)
            if v < obstacle_dist:
                obstacle_dist = v

        with self._lock:
            self._scan_ranges = normalized
            self._obstacle_distance = obstacle_dist * LIDAR_DISTANCE_CAP

    def _on_odom(self, sample: zenoh.Sample) -> None:
        """Deserialize Odometry CDR, extract position and heading, cache."""
        try:
            msg = Odometry.deserialize(bytes(sample.payload))
        except Exception as exc:
            print(f"[odom] CDR deserialize error: {exc}")
            return

        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(ori.x, ori.y, ori.z, ori.w)

        with self._lock:
            self._robot_x = pos.x
            self._robot_y = pos.y
            self._robot_heading = yaw
            self._recompute_goal_vector()

    def _on_goal_pose(self, sample: zenoh.Sample) -> None:
        """Deserialize Pose CDR, extract goal position, cache."""
        try:
            msg = Pose.deserialize(bytes(sample.payload))
        except Exception as exc:
            print(f"[goal_pose] CDR deserialize error: {exc}")
            return

        with self._lock:
            self._goal_x = msg.position.x
            self._goal_y = msg.position.y
            self._new_goal = True
            self._recompute_goal_vector()
        print(f"[goal_pose] new goal: x={msg.position.x:.2f} y={msg.position.y:.2f}")

    def _on_step_response(self, sample: zenoh.Sample) -> None:
        """Receive step response JSON from the environment node."""
        try:
            payload = json.loads(bytes(sample.payload).decode())
        except Exception as exc:
            print(f"[step_response] JSON decode error: {exc}")
            return
        self._step_response = payload
        self._step_event.set()

    # ----------------------------------------------------------------- #
    #                      Internal Helpers                              #
    # ----------------------------------------------------------------- #

    def _recompute_goal_vector(self) -> None:
        """Recompute goal distance and angle from cached robot/goal poses.

        Must be called while ``self._lock`` is held.
        """
        diff_x = self._goal_x - self._robot_x
        diff_y = self._goal_y - self._robot_y
        distance = math.sqrt(diff_x ** 2 + diff_y ** 2)
        heading_to_goal = math.atan2(diff_y, diff_x)
        angle = heading_to_goal - self._robot_heading

        # Normalize angle to [-pi, pi]
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi

        self._goal_distance = distance
        self._goal_angle = angle

    # ----------------------------------------------------------------- #
    #                         Public API                                 #
    # ----------------------------------------------------------------- #

    def publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Publish a Twist velocity command via Zenoh (CDR-encoded).

        Parameters
        ----------
        linear_x : float
            Forward/backward velocity in m/s.
        angular_z : float
            Rotational velocity in rad/s.
        """
        twist = Twist(
            linear=Vector3(x=linear_x, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=angular_z),
        )
        self._pub_cmd_vel.put(twist.serialize())

    def publish_metrics(self, metrics: Dict) -> None:
        """Publish training metrics as JSON.

        Parameters
        ----------
        metrics : dict
            Arbitrary metric payload (e.g. episode, reward, loss).
        """
        self._pub_metrics.put(json.dumps(metrics).encode())

    def get_state(
        self,
        action_linear_previous: float = 0.0,
        action_angular_previous: float = 0.0,
    ) -> List[float]:
        """Build the 44-dimensional state vector from cached sensor data.

        Layout (matches DRLEnvironment.get_state):
            [0..39]  - 40 normalized LiDAR samples     (range [0, 1])
            [40]     - normalized goal distance         (range [0, 1])
            [41]     - normalized goal angle            (range [-1, 1])
            [42]     - previous linear action           (range [-1, 1])
            [43]     - previous angular action          (range [-1, 1])

        Parameters
        ----------
        action_linear_previous : float
            Normalized linear action from the previous step.
        action_angular_previous : float
            Normalized angular action from the previous step.

        Returns
        -------
        list[float]
            State vector of length ``num_scan_samples + 4``.
        """
        with self._lock:
            state = list(self._scan_ranges)
            state.append(float(np.clip(self._goal_distance / MAX_GOAL_DISTANCE, 0.0, 1.0)))
            state.append(float(self._goal_angle) / math.pi)
        state.append(float(action_linear_previous))
        state.append(float(action_angular_previous))
        return state

    @property
    def goal_distance(self) -> float:
        """Current Euclidean distance to the goal (metres)."""
        with self._lock:
            return self._goal_distance

    @property
    def obstacle_distance(self) -> float:
        """Distance to the closest detected obstacle (metres)."""
        with self._lock:
            return self._obstacle_distance

    @property
    def robot_position(self) -> Tuple[float, float]:
        """Current robot (x, y) position in world frame."""
        with self._lock:
            return self._robot_x, self._robot_y

    @property
    def goal_position(self) -> Tuple[float, float]:
        """Current goal (x, y) position."""
        with self._lock:
            return self._goal_x, self._goal_y

    @property
    def new_goal(self) -> bool:
        """Whether a new goal has been received since the last check."""
        with self._lock:
            return self._new_goal

    def clear_new_goal(self) -> None:
        """Acknowledge receipt of the new goal."""
        with self._lock:
            self._new_goal = False

    def wait_for_goal(self, poll_interval: float = 1.0) -> None:
        """Block until a new goal pose is received.

        Parameters
        ----------
        poll_interval : float
            Seconds between polling checks.
        """
        while not self.new_goal:
            print("Waiting for new goal...")
            time.sleep(poll_interval)

    # ----------------------------------------------------------------- #
    #                     Step Synchronization                           #
    # ----------------------------------------------------------------- #

    def step(
        self,
        action: List[float],
        previous_action: List[float],
    ) -> Tuple[List[float], float, bool, int, float]:
        """Execute one environment step via Zenoh.

        The method:
            1. Un-normalizes and publishes the velocity command via ``rt/cmd_vel``.
            2. Publishes a step request JSON to ``tb/drl/step_request``.
            3. Waits for the environment node to respond on ``tb/drl/step_response``.
            4. Returns the transition tuple.

        Parameters
        ----------
        action : list[float]
            Normalized action ``[linear, angular]``. Pass an empty list to
            initialize an episode (the environment returns the initial state
            with zero reward).
        previous_action : list[float]
            Normalized action from the previous step ``[linear, angular]``.

        Returns
        -------
        tuple
            ``(state, reward, done, outcome, distance_traveled)`` mirroring
            the ``DrlStep.Response`` fields.

        Raises
        ------
        TimeoutError
            If the environment does not respond within ``step_timeout``.
        """
        # Publish velocity command if we have a real action
        if len(action) > 0:
            linear_norm = action[0]
            angular_norm = action[1]

            if self.enable_backward:
                linear_vel = linear_norm * SPEED_LINEAR_MAX
            else:
                linear_vel = (linear_norm + 1.0) / 2.0 * SPEED_LINEAR_MAX
            angular_vel = angular_norm * SPEED_ANGULAR_MAX

            self.publish_cmd_vel(linear_vel, angular_vel)
        else:
            # Episode init: stop the robot
            self.publish_cmd_vel(0.0, 0.0)

        # Build and publish step request
        request = {
            "action": action,
            "previous_action": previous_action,
        }
        self._step_event.clear()
        self._step_response = None
        self._pub_step_request.put(json.dumps(request).encode())

        # Wait for response from the environment node
        if not self._step_event.wait(timeout=self.step_timeout):
            raise TimeoutError(
                f"No step response received within {self.step_timeout}s. "
                "Is the environment node running?"
            )

        resp = self._step_response
        if resp is None:
            raise RuntimeError("Step response was None despite event being set.")

        state = resp.get("state", [])
        reward = float(resp.get("reward", 0.0))
        done = bool(resp.get("done", False))
        outcome = int(resp.get("success", 0))
        distance_traveled = float(resp.get("distance_traveled", 0.0))

        return state, reward, done, outcome, distance_traveled

    def init_episode(self) -> List[float]:
        """Initialize a new episode and return the initial state.

        Sends an empty action to the environment, which resets episode
        variables and returns the starting observation.
        """
        state, _, _, _, _ = self.step(action=[], previous_action=[0.0, 0.0])
        return state

    # ----------------------------------------------------------------- #
    #                          Lifecycle                                 #
    # ----------------------------------------------------------------- #

    def close(self) -> None:
        """Cleanly shut down all Zenoh resources."""
        print("ZenohDRLAdapter shutting down...")

        # Stop the robot
        self.publish_cmd_vel(0.0, 0.0)

        # Undeclare subscribers
        self._sub_scan.undeclare()
        self._sub_odom.undeclare()
        self._sub_goal.undeclare()
        self._sub_step_response.undeclare()

        # Undeclare publishers
        self._pub_cmd_vel.undeclare()
        self._pub_step_request.undeclare()
        self._pub_metrics.undeclare()

        # Close session
        self._session.close()
        print("ZenohDRLAdapter stopped.")

    def __enter__(self) -> "ZenohDRLAdapter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
