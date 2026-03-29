TurtleBot3 Maze - Object Detection Pipeline
=============================================

Alula Gebreegziabher
CS 485 - AI, NJIT, Spring 2026
Assignment 2: Object Detection Events with ROS 2 + Zenoh + PostgreSQL

This builds on Assignment 1 by adding a camera to the TurtleBot3 and
running YOLOv8 to detect objects placed on benches in the maze.
Detections go through Zenoh and get stored in PostgreSQL.

Setup
-----
Make sure Docker and Docker Compose are installed.

Build everything:
  docker compose build

Run the full pipeline:
  docker compose up simulation postgres zenoh detection ingest

What it does
------------
1. Gazebo simulation runs the TurtleBot3 in the maze
2. Detection node subscribes to /camera/image_raw, runs YOLO
3. Detection events are published to Zenoh
4. Ingest worker picks them up and writes to PostgreSQL

Check the database:
  docker exec -it a2-postgres psql -U postgres -d detections
  SELECT class_name, count(*) FROM detections GROUP BY class_name;

Files
-----
detection_node/detector.py  - ROS 2 node that runs YOLO and publishes to Zenoh
zenoh_ingest/ingest.py      - subscribes to Zenoh, writes to PostgreSQL
sql/schema.sql              - database tables
world_assets.md             - bench positions and object lists
