# Run Report

## Setup
- ROS 2 Humble + Gazebo Classic
- TurtleBot3 Burger with camera
- YOLOv8n for detection
- Zenoh for messaging
- PostgreSQL for storage

## How to run
docker compose build
docker compose up simulation postgres zenoh detection ingest

## Checking results
docker exec -it a2-postgres psql -U postgres -d detections

-- how many events
SELECT count(*) FROM detection_events;

-- how many detections
SELECT count(*) FROM detections;

-- what was detected
SELECT class_name, count(*) as cnt
FROM detections
GROUP BY class_name
ORDER BY cnt DESC;
