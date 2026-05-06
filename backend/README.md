# Perception Demo UI (Backend)

FastAPI service that runs the YOLOX + SORT + Anomaly Detection tracking pipeline and streams frames to the frontend over a WebSocket endpoint.

## Requirements

- Python 3.10+
- Dependencies: fastapi, uvicorn, pydantic, numpy, onnxruntime,
  opencv-python

## Run

From the repo root:

- python -m uvicorn backend.anomaly_api:app --host 0.0.0.0 --port 8000

## Configuration

Defaults live in PipelineConfig inside anomaly_api.py. Common fields:

- model_path / prototxt_path / artifacts_folder
- predictor_model_path / predictor_window_size / anomaly_threshold
- data_source: "nuscenes" or "video"
- root_dir: dataset root (nuScenes or video frame folders)
- cameras: list of CAM_* names (nuscenes) or subfolder names (video)
- sort_* and threshold parameters

Note: model paths are treated as immutable at runtime; change them and restart the
server.

## Data sources

### NuScenes

Set data_source="nuscenes" and root_dir to a local nuScenes dataset. Cameras
should be CAM_* names (e.g., CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT).

### Video frames

Set data_source="video" and root_dir to a directory with one subfolder per
camera/video stream. Example layout:

root_dir/
  front/
    0000.jpg
    0001.jpg
  front_left/
    0000.jpg

Each subfolder is treated as a synchronized stream by frame index.

## API

- GET /health: service status and subscriber counts
- GET /config: current pipeline configuration
- WS  /tracking: streamed frames

WebSocket frame format (object tracks):
```json
{
  "cam_id": "CAM_FRONT",
  "metadata": { ... },
  "tracks": {
    "42": { "bbox": [x1, y1, x2, y2], "score": 0.85, "class_id": 0, "anomaly_score": 0.12 }
  },
  "timings": { "detection_ms": 12.3, ... },
  "image": "base64..."
}
```