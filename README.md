# Perception demo UI

Vanilla **HTML / CSS / JS**: six camera tiles (placeholders) with optional detection overlays. **`CAM_FRONT_sample.json`** in this folder is used for **offline playback** on the **front** camera only.

---

## Run with the bundled `CAM_FRONT_sample.json`

1. **Open a terminal** in your editor (**Terminal → New Terminal** in VS Code/Cursor) or **Terminal.app** on macOS.

2. **Go to this project folder** (change the path if yours differs):
   ```bash
   cd ~/Desktop/perception-demo-ui
   ```
   Run `ls` and confirm **`CAM_FRONT_sample.json`** is listed.

3. **Start a local web server:**
   ```bash
   python3 -m http.server 8080
   ```
   Keep this window open. Stop with **Ctrl+C** when done.

4. **In your browser**, open:
   ```
   http://127.0.0.1:8080/?playback=1
   ```
   **`?playback=1`** loads `CAM_FRONT_sample.json` and steps through frames. No WebSocket or SSH tunnel needed.

5. On the page, use **“Show object detection”** to toggle boxes. Only the **front** tile uses this JSON; other cameras stay placeholders.

**Note:** Don’t open `index.html` via **File → Open** for playback—a `file://` URL usually breaks **`fetch`**. Use steps 3–4.

---

## Requirements

- **Python 3**
- Same folder: `index.html`, `main.js`, `CAM_FRONT_sample.json`

---

## Optional: live WebSocket

Use **`http://127.0.0.1:8080/`** (no `?playback`) when the backend is running. The UI uses **`ws://127.0.0.1:8001/tracking`** by default (often with an SSH tunnel). That’s separate from JSON playback.

---

## Notes

- **Front** box coordinates are scaled from **1600×900** annotation space to the image on screen; adjust **`ANNOTATION_SIZE_BY_CAMERA`** in `main.js` if your data uses a different size.
