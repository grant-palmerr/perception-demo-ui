# Perception Demo UI

Real-time anomaly tracking demo with a FastAPI backend streaming detections to a
static HTML/JS frontend.

## Project layout

- backend/ — FastAPI app, dataloaders, tracking pipeline, models
  - Designed to run on the TI TDA4VM proccessor
- frontend/ — static UI (index.html, main.js, worker.js)
  - Can be ran locally or on the proccessor

## Quick start

1. Start the backend (see backend/README.md for details).
2. Serve the frontend (see frontend/README.md for details).
3. Open the UI and connect to ws://127.0.0.1:8000/tracking.

## Data sources

- NuScenes (default): update root_dir and cameras in backend/anomaly_api.py.
- Video frames: set data_source="video" and point root_dir to a folder that
	contains one subfolder per camera (each with numbered .jpg/.png frames).

## More docs

- Backend details: backend/README.md
- Frontend details: frontend/README.md

## Model Repository
https://github.com/WilliamFlinchbaugh/traffic-tracking-anomaly-detection