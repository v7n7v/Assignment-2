"""
Zenoh bridge adapter for turtlebot3_drlnav.

Provides ROS-free communication between the DRL agent container and the
ROS 2 simulation environment via Zenoh pub/sub with pycdr2 CDR
serialization.

Usage::

    from turtlebot3_drl.zenoh_bridge import ZenohDRLAdapter

    adapter = ZenohDRLAdapter(connect="tcp/192.168.1.10:7447")
    state = adapter.init_episode()
    next_state, reward, done, outcome, dist = adapter.step(action, prev_action)
    adapter.close()
"""

from .cdr_types import (
    Header,
    LaserScan,
    Odometry,
    Point,
    Pose,
    Quaternion,
    Time,
    Twist,
    Vector3,
)
from .zenoh_adapter import ZenohDRLAdapter

__all__ = [
    "ZenohDRLAdapter",
    "Header",
    "LaserScan",
    "Odometry",
    "Point",
    "Pose",
    "Quaternion",
    "Time",
    "Twist",
    "Vector3",
]
