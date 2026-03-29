#!/usr/bin/env python3
"""Subscribes to the TurtleBot camera, runs YOLO, publishes detections to Zenoh."""

import json
import math
import hashlib
import uuid
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
import numpy as np
from ultralytics import YOLO

import zenoh


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.model = YOLO('yolov8n.pt')
        self.run_id = str(uuid.uuid4())
        self.robot_id = 'tb3_sim'
        self.sequence = 0

        self.latest_odom = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)

        # open zenoh session
        conf = zenoh.Config()
        self.session = zenoh.open(conf)
        self.get_logger().info(f'started, run_id={self.run_id}')

        # publish run metadata once
        meta_key = f'maze/{self.robot_id}/{self.run_id}/runmeta/v1'
        meta = json.dumps({
            'run_id': self.run_id,
            'robot_id': self.robot_id,
            'start_time': time.time()
        })
        self.session.put(meta_key, meta.encode())

    def odom_cb(self, msg):
        self.latest_odom = msg

    def image_cb(self, msg):
        if self.latest_odom is None:
            return

        # convert ros image to numpy array
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        img_hash = hashlib.sha256(msg.data).hexdigest()

        # run yolo
        results = self.model(img, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                dets.append({
                    'det_id': str(uuid.uuid4()),
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])],
                    'confidence': round(float(box.conf[0]), 4),
                    'bbox_xyxy': [round(float(c), 2) for c in box.xyxy[0]]
                })

        if not dets:
            return

        # grab odometry
        odom = self.latest_odom
        pos = odom.pose.pose.position
        orient = odom.pose.pose.orientation
        siny = 2.0 * (orient.w * orient.z + orient.x * orient.y)
        cosy = 1.0 - 2.0 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny, cosy)

        # try to get base->camera transform
        tf_ok = False
        t_base_camera = [0.0] * 16
        try:
            t = self.tf_buffer.lookup_transform(
                'base_footprint', 'camera_link', rclpy.time.Time()
            )
            tf_ok = True
            tr = t.transform.translation
            t_base_camera = [1,0,0,tr.x, 0,1,0,tr.y, 0,0,1,tr.z, 0,0,0,1]
        except Exception:
            pass

        event_id = str(uuid.uuid4())
        self.sequence += 1

        # build the event json (follows maze.detection.v1 schema)
        event = {
            'schema': 'maze.detection.v1',
            'event_id': event_id,
            'run_id': self.run_id,
            'robot_id': self.robot_id,
            'sequence': self.sequence,
            'image': {
                'topic': '/camera/image_raw',
                'stamp': {
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec
                },
                'frame_id': msg.header.frame_id,
                'width': msg.width,
                'height': msg.height,
                'encoding': msg.encoding,
                'sha256': img_hash
            },
            'odometry': {
                'topic': '/odom',
                'frame_id': odom.header.frame_id,
                'x': round(pos.x, 4),
                'y': round(pos.y, 4),
                'yaw': round(yaw, 4),
                'vx': round(odom.twist.twist.linear.x, 4),
                'vy': round(odom.twist.twist.linear.y, 4),
                'wz': round(odom.twist.twist.angular.z, 4)
            },
            'tf': {
                'base_frame': 'base_footprint',
                'camera_frame': 'camera_link',
                't_base_camera': t_base_camera,
                'tf_ok': tf_ok
            },
            'detections': dets
        }

        key = f'maze/{self.robot_id}/{self.run_id}/detections/v1/{event_id}'
        self.session.put(key, json.dumps(event).encode())
        self.get_logger().info(
            f'seq={self.sequence} detections={len(dets)}'
        )


def main():
    rclpy.init()
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.session.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
