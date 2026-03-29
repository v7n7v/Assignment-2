CREATE TABLE detection_events (
    event_id       uuid PRIMARY KEY,
    run_id         uuid NOT NULL,
    robot_id       text NOT NULL,
    sequence       bigint NOT NULL,
    stamp          timestamptz NOT NULL,
    image_frame_id text,
    image_sha256   text,
    width          int,
    height         int,
    encoding       text,
    x              double precision,
    y              double precision,
    yaw            double precision,
    vx             double precision,
    vy             double precision,
    wz             double precision,
    tf_ok          boolean,
    t_base_camera  double precision[],
    raw_event      jsonb NOT NULL,
    UNIQUE(run_id, robot_id, sequence)
);

CREATE TABLE detections (
    det_pk     bigserial PRIMARY KEY,
    event_id   uuid REFERENCES detection_events(event_id) ON DELETE CASCADE,
    det_id     uuid,
    class_id   int,
    class_name text,
    confidence double precision,
    x1         double precision,
    y1         double precision,
    x2         double precision,
    y2         double precision,
    UNIQUE(event_id, det_id)
);
