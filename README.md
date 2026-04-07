# Perception demo UI

Little **HTML/CSS/JS** thing for class — six fake camera tiles, green boxes when we have detections. The **`CAM_FRONT_sample.json`** in the repo is only for **offline playback** on the **front** cam.

---

## Run it with the sample json

1. **Open a terminal** (Cursor/VS Code terminal is fine, or Terminal.app).

2. **cd into the repo:**
   ```bash
   cd ~/Desktop/perception-demo-ui
   ```
   Run `ls` and make sure **`CAM_FRONT_sample.json`** is there.

3. **Tiny http server** (Python is already on Macs usually):
   ```bash
   python3 -m http.server 8080
   ```
   Leave it running. Ctrl+C to stop.

4. **Browser:**
   ```
   http://127.0.0.1:8080/?playback=1
   ```
   The **`?playback=1`** bit matters — that’s what loads the json and flips through frames. No ssh tunnel needed for this.

5. Uncheck **“Show object detection”** if you just want the placeholders. Only **front** gets real tracks from this file right now.

**Don’t** open `index.html` straight from Finder for playback — **`fetch` dies on file://** most of the time.

---

## What you need

- Python 3  
- These files together: `index.html`, `main.js`, `CAM_FRONT_sample.json`

---

## Live WebSocket mode (optional)

`http://127.0.0.1:8080/` **without** `?playback` — talks to **`ws://127.0.0.1:8001/tracking`** (tunnel + backend situation). Different from the json playback thing.

---

## Random note

Front camera boxes get scaled from **1600×900** idea into whatever image size we show — mess with **`ANNOTATION_SIZE_BY_CAMERA`** in `main.js` if your data isn’t nuScenes-shaped.
