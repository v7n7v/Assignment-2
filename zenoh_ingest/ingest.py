#!/usr/bin/env python3
"""Subscribes to Zenoh detection events and writes them to PostgreSQL."""

import json
import os
import time
from datetime import datetime, timezone

import psycopg2
import zenoh


DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'detections')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASS = os.environ.get('DB_PASS', 'postgres')


def get_conn():
    """Keep retrying until postgres is ready."""
    for attempt in range(30):
        try:
            return psycopg2.connect(
                host=DB_HOST, port=DB_PORT,
                dbname=DB_NAME, user=DB_USER, password=DB_PASS
            )
        except psycopg2.OperationalError:
            print(f'waiting for db (attempt {attempt + 1})')
            time.sleep(2)
    raise RuntimeError('could not connect to database')


def insert_event(conn, event):
    stamp = event['image']['stamp']
    ts = datetime.fromtimestamp(
        stamp['sec'] + stamp['nanosec'] / 1e9, tz=timezone.utc
    )

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO detection_events
                (event_id, run_id, robot_id, sequence, stamp,
                 image_frame_id, image_sha256, width, height, encoding,
                 x, y, yaw, vx, vy, wz, tf_ok, t_base_camera, raw_event)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """, (
                event['event_id'], event['run_id'], event['robot_id'],
                event['sequence'], ts,
                event['image'].get('frame_id'),
                event['image'].get('sha256'),
                event['image'].get('width'),
                event['image'].get('height'),
                event['image'].get('encoding'),
                event['odometry']['x'], event['odometry']['y'],
                event['odometry']['yaw'],
                event['odometry']['vx'], event['odometry']['vy'],
                event['odometry']['wz'],
                event['tf']['tf_ok'], event['tf']['t_base_camera'],
                json.dumps(event)
            ))

            for det in event.get('detections', []):
                bbox = det['bbox_xyxy']
                cur.execute("""
                    INSERT INTO detections
                    (event_id, det_id, class_id, class_name, confidence,
                     x1, y1, x2, y2)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                """, (
                    event['event_id'], det['det_id'],
                    det['class_id'], det['class_name'], det['confidence'],
                    bbox[0], bbox[1], bbox[2], bbox[3]
                ))

            conn.commit()
            print(f"inserted event seq={event['sequence']} "
                  f"({len(event.get('detections', []))} dets)")
    except Exception as e:
        conn.rollback()
        print(f'error: {e}')


def main():
    conn = get_conn()
    print('connected to db')

    conf = zenoh.Config()
    session = zenoh.open(conf)
    print('subscribing to maze/**/detections/v1/*')

    def on_sample(sample):
        try:
            event = json.loads(bytes(sample.payload))
            insert_event(conn, event)
        except Exception as e:
            print(f'bad sample: {e}')

    sub = session.declare_subscriber('maze/**/detections/v1/*', on_sample)
    print('running...')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    sub.undeclare()
    session.close()
    conn.close()


if __name__ == '__main__':
    main()
