# Perception Demo UI (Frontend)

Static HTML/CSS/JS UI that renders camera tiles and overlays detections received
over a WebSocket stream.

## Run locally

1. Start a static server in this folder.
2. Open http://127.0.0.1:8080/ in a browser.
3. Click “Connect” and confirm the WebSocket URL (default is
   ws://127.0.0.1:8000/tracking).

## Configuration

- Default WebSocket URL: edit DEFAULT_WEBSOCKET_URL in main.js.
- The UI stores the WebSocket URL in localStorage (key: wsUrl).
- Object class names are defined in worker.js (CLASS_NAMES).

## Backend expectation

The UI expects frames from the /tracking WebSocket with:

- cam_id
- metadata
- tracks (object format with bbox/score/class_id/anomaly_score)
- image (base64-encoded JPEG)

See backend/README.md for the full API details.
