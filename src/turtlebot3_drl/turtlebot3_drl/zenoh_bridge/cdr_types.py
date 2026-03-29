"""
CDR (Common Data Representation) type definitions for ROS 2 messages.

Defines pycdr2 IdlStruct dataclasses that mirror the ROS 2 message types used
by the DRL navigation system. These allow CDR serialization/deserialization
WITHOUT any ROS 2 dependency, communicating via zenoh-bridge-ros2dds.

Message types:
    - sensor_msgs/msg/LaserScan  (LiDAR data)
    - nav_msgs/msg/Odometry      (robot pose and velocity)
    - geometry_msgs/msg/Twist     (velocity commands)
    - geometry_msgs/msg/Pose      (goal position)

Based on the pycdr2 pattern from:
    turtlebot-maze/detector/object_detector.py
    turtlebot-maze/slam/slam_bridge.py
"""

from dataclasses import dataclass, field
from typing import List

from pycdr2 import IdlStruct
from pycdr2.types import float32, float64, int32, uint32, uint8


# ===================================================================== #
#                        Primitive / Shared Types                        #
# ===================================================================== #

@dataclass
class Time(IdlStruct, typename="builtin_interfaces/msg/Time"):
    """builtin_interfaces/msg/Time"""
    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    """std_msgs/msg/Header"""
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class Vector3(IdlStruct, typename="geometry_msgs/msg/Vector3"):
    """geometry_msgs/msg/Vector3 (float64 fields)."""
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Point(IdlStruct, typename="geometry_msgs/msg/Point"):
    """geometry_msgs/msg/Point (float64 fields)."""
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs/msg/Quaternion"):
    """geometry_msgs/msg/Quaternion (float64 fields)."""
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0
    w: float64 = 1.0


# ===================================================================== #
#                        geometry_msgs Types                             #
# ===================================================================== #

@dataclass
class Pose(IdlStruct, typename="geometry_msgs/msg/Pose"):
    """geometry_msgs/msg/Pose"""
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseWithCovariance(IdlStruct, typename="geometry_msgs/msg/PoseWithCovariance"):
    """geometry_msgs/msg/PoseWithCovariance"""
    pose: Pose = field(default_factory=Pose)
    covariance: List[float64] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class Twist(IdlStruct, typename="geometry_msgs/msg/Twist"):
    """geometry_msgs/msg/Twist"""
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


@dataclass
class TwistWithCovariance(IdlStruct, typename="geometry_msgs/msg/TwistWithCovariance"):
    """geometry_msgs/msg/TwistWithCovariance"""
    twist: Twist = field(default_factory=Twist)
    covariance: List[float64] = field(default_factory=lambda: [0.0] * 36)


# ===================================================================== #
#                         sensor_msgs Types                              #
# ===================================================================== #

@dataclass
class LaserScan(IdlStruct, typename="sensor_msgs/msg/LaserScan"):
    """sensor_msgs/msg/LaserScan

    See: https://docs.ros2.org/latest/api/sensor_msgs/msg/LaserScan.html
    """
    header: Header = field(default_factory=Header)
    angle_min: float32 = 0.0
    angle_max: float32 = 0.0
    angle_increment: float32 = 0.0
    time_increment: float32 = 0.0
    scan_time: float32 = 0.0
    range_min: float32 = 0.0
    range_max: float32 = 0.0
    ranges: List[float32] = field(default_factory=list)
    intensities: List[float32] = field(default_factory=list)


# ===================================================================== #
#                          nav_msgs Types                                #
# ===================================================================== #

@dataclass
class Odometry(IdlStruct, typename="nav_msgs/msg/Odometry"):
    """nav_msgs/msg/Odometry

    See: https://docs.ros2.org/latest/api/nav_msgs/msg/Odometry.html
    """
    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)
    twist: TwistWithCovariance = field(default_factory=TwistWithCovariance)
