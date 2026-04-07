/**
 * Perception demo – Step 1: WebSocket connection and message handling.
 * Step 2 will add drawing/canvas in #viewer.
 */

// WebSocket endpoint for this demo (adjust as needed).
const WEBSOCKET_URL = "ws://127.0.0.1:8001/tracking";

/** Local sample playback: open with ?playback=1 (served via http.server, not file://). */
const SAMPLE_JSON_PATH = "CAM_FRONT_sample.json";
const PLAYBACK_INTERVAL_MS = 150;

function getPlaybackModeFromUrl() {
  const p = new URLSearchParams(window.location.search).get("playback");
  return p === "1" || p === "true" || p === "yes";
}

const IS_PLAYBACK_MODE = getPlaybackModeFromUrl();

/**
 * class_id → label. Sync with tracking_pipeline.py on the board when known.
 * Sample data uses 0, 1, … — adjust names as needed.
 */
const CLASS_NAMES = [
  "car",
  "pedestrian"
];

/** Playback-only: frame list + timer (WebSocket not used). */
const playbackState = {
  frames: null,
  totalFrames: 0,
  frameIndex: 0,
  timerId: null,
  frameLabel: "—"
};

// Static camera images. Email URLs returned 404; using placeholders until correct URLs are provided.
// Original email URLs (for reference): smuseniordesign.blob.core.windows.net/nuimages/CAM FRONT/...
const CAMERA_IMAGES = {
  front: "https://placehold.co/640x480/2a2a32/b8b8c8?text=front",
  front_left: "https://placehold.co/640x480/2a2a32/b8b8c8?text=front_left",
  front_right: "https://placehold.co/640x480/2a2a32/b8b8c8?text=front_right",
  back: "https://placehold.co/640x480/2a2a32/b8b8c8?text=back",
  back_left: "https://placehold.co/640x480/2a2a32/b8b8c8?text=back_left",
  back_right: "https://placehold.co/640x480/2a2a32/b8b8c8?text=back_right"
};

/**
 * Pixel space of bounding_box data from the tracker / dataset (before display).
 * nuScenes CAM_FRONT images are 1600×900 — matches CAM_FRONT_sample.json tracks.
 * Scale: canvas_coord = annotation_coord * (canvasSize / annotationSize).
 * Omit a camera (or use w/h 0) to draw boxes 1:1 with the loaded image (e.g. mock WebSocket in image space).
 */
const ANNOTATION_SIZE_BY_CAMERA = {
  front: { w: 1600, h: 900 }
};

/**
 * @returns {{ sx: number, sy: number }}
 */
function getAnnotationScale(cameraId, canvasW, canvasH) {
  const ref = ANNOTATION_SIZE_BY_CAMERA[cameraId];
  if (!ref || !ref.w || !ref.h || ref.w <= 0 || ref.h <= 0) {
    return { sx: 1, sy: 1 };
  }
  return { sx: canvasW / ref.w, sy: canvasH / ref.h };
}

const appState = {
  connectionStatus: "disconnected", // "disconnected" | "connecting" | "connected" | "error"
  lastMessage: null,                // object | null
  lastMessageReceivedAt: null,      // timestamp number | null
  messageCount: 0,                  // number of valid messages received
  showDetections: true               // when false, hide bounding boxes and labels
};

const statusHeader = document.getElementById("status-header");

function getDetections(msg) {
  if (!msg) return [];
  if (Array.isArray(msg) && msg.length > 0 && typeof msg[0] === "object" && "bounding_box" in (msg[0] ?? {})) {
    return msg;
  }
  if (Array.isArray(msg)) return msg;
  return msg.detections ?? msg.objects ?? msg.boxes ?? [];
}

/**
 * Map metadata.camera (e.g. CAM_FRONT) to UI dataset.camera ids (e.g. front).
 */
function cameraIdFromMetadata(camera) {
  const key = String(camera || "")
    .toUpperCase()
    .replace(/\s+/g, "_")
    .replace(/^CAM_/, "");
  const map = {
    FRONT: "front",
    FRONT_LEFT: "front_left",
    FRONT_RIGHT: "front_right",
    BACK: "back",
    BACK_LEFT: "back_left",
    BACK_RIGHT: "back_right"
  };
  return map[key] || "front";
}

/**
 * One frame from CAM_FRONT_sample.json → detections array for renderViewer.
 */
function frameToDetections(frame) {
  const meta = frame && frame.metadata ? frame.metadata : {};
  const cam = cameraIdFromMetadata(meta.camera);
  const tracks = frame && frame.tracks ? frame.tracks : {};
  const out = [];

  for (const [trackId, arr] of Object.entries(tracks)) {
    if (!Array.isArray(arr) || arr.length < 6) continue;
    const x1 = Number(arr[0]);
    const y1 = Number(arr[1]);
    const x2 = Number(arr[2]);
    const y2 = Number(arr[3]);
    const conf = Number(arr[4]);
    const classId = Math.floor(Number(arr[5]));
    const objectClass = CLASS_NAMES[classId] ?? `class_${classId}`;

    out.push({
      camera_id: cam,
      object_class: objectClass,
      bounding_box: [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
      ],
      track_id: trackId,
      confidence: conf
    });
  }
  return out;
}

function drawBoundingBox(ctx, d, sx, sy) {
  const scaleX = sx != null && sx > 0 ? sx : 1;
  const scaleY = sy != null && sy > 0 ? sy : 1;
  const bbox = d.bounding_box ?? d.bbox ?? d.box;
  if (!Array.isArray(bbox) || bbox.length < 4) return;
  const points = bbox
    .map((p) => (Array.isArray(p) ? [parseFloat(p[0]), parseFloat(p[1])] : null))
    .filter(Boolean)
    .map(([x, y]) => [x * scaleX, y * scaleY]);
  if (points.length !== 4) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
  ctx.stroke();
  if (d.object_class) {
    ctx.fillStyle = "lime";
    const fontPx = Math.max(10, Math.round(12 * Math.min(scaleX, scaleY)));
    ctx.font = fontPx + "px sans-serif";
    ctx.fillText(d.object_class, points[0][0], points[0][1] - 4);
  }
}

function renderViewer(message) {
  if (!message) return;
  const detections = getDetections(message);
  const slots = document.querySelectorAll(".camera-slot");

  slots.forEach((slot) => {
    const cameraId = slot.dataset.camera;
    const img = slot.querySelector("img");
    const canvas = slot.querySelector("canvas");
    if (!img || !canvas || !CAMERA_IMAGES[cameraId]) return;

    const cameraDetections = detections.filter(
      (d) => (d.camera_id || "").toLowerCase().replace(/\s/g, "_") === cameraId
    );

    const drawBoxes = () => {
      if (!img.naturalWidth) return;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!appState.showDetections) return;
      if (cameraDetections.length > 0) {
        const { sx, sy } = getAnnotationScale(cameraId, canvas.width, canvas.height);
        ctx.strokeStyle = "lime";
        ctx.lineWidth = Math.max(1, Math.min(2, 2 * Math.min(sx, sy)));
        for (const d of cameraDetections) drawBoundingBox(ctx, d, sx, sy);
      }
    };

    if (img.src !== CAMERA_IMAGES[cameraId]) {
      img.src = CAMERA_IMAGES[cameraId];
      img.onload = drawBoxes;
    } else if (img.complete) {
      drawBoxes();
    }
  });
}

function updateStatusUI() {
  if (!statusHeader) return;

  if (IS_PLAYBACK_MODE) {
    statusHeader.dataset.status =
      appState.connectionStatus === "playback" ? "playback" : appState.connectionStatus;
    if (appState.connectionStatus === "connecting") {
      statusHeader.innerHTML =
        '<span class="status-row">' +
        '<span class="kv"><strong>Mode</strong> <span class="val">Local playback</span></span>' +
        '<span class="url">(loading ' +
        SAMPLE_JSON_PATH +
        "…)</span>" +
        "</span>";
    } else if (appState.connectionStatus === "playback") {
      statusHeader.innerHTML =
        '<span class="status-row">' +
        '<span class="kv"><strong>Mode</strong> <span class="val">Local playback</span></span>' +
        '<span class="url">(' +
        SAMPLE_JSON_PATH +
        ")</span>" +
        '<span class="kv"><strong>Frame</strong> <span class="val">' +
        playbackState.frameLabel +
        "</span></span>" +
        '<span class="kv"><strong>Steps</strong> <span class="val">' +
        appState.messageCount +
        "</span></span>" +
        "</span>";
    } else if (appState.connectionStatus === "error") {
      statusHeader.innerHTML =
        '<span class="status-row">' +
        '<span class="kv"><strong>Playback</strong> <span class="val">failed</span></span>' +
        '<span class="url">(check console; serve with python3 -m http.server and use ' +
        SAMPLE_JSON_PATH +
        " in project folder)</span>" +
        "</span>";
    } else {
      statusHeader.innerHTML =
        '<span class="status-row">' +
        '<span class="kv"><strong>Mode</strong> <span class="val">' +
        appState.connectionStatus +
        "</span></span></span>";
    }
    return;
  }

  statusHeader.dataset.status = appState.connectionStatus;
  statusHeader.innerHTML =
    '<span class="status-row">' +
    '<span class="kv"><strong>Connection</strong> <span class="val">' +
    appState.connectionStatus +
    "</span></span>" +
    '<span class="url">(' +
    WEBSOCKET_URL +
    ")</span>" +
    '<span class="kv"><strong>Messages</strong> <span class="val">' +
    appState.messageCount +
    "</span></span>" +
    "</span>";
}

/**
 * Connect to the WebSocket.
 * Uses the global WEBSOCKET_URL constant.
 */
function connectWebSocket() {
  appState.connectionStatus = "connecting";
  updateStatusUI();

  const ws = new WebSocket(WEBSOCKET_URL);

  ws.onopen = () => {
    appState.connectionStatus = "connected";
    updateStatusUI();
    console.log("WebSocket connected to", WEBSOCKET_URL);
  };

  ws.onmessage = (event) => {
    let parsed;
    try {
      parsed = JSON.parse(event.data);
    } catch (e) {
      console.error("WebSocket message: invalid JSON", e.message);
      return;
    }
    if (typeof parsed !== "object" || parsed === null) {
      console.warn("WebSocket message: expected object, got", typeof parsed);
      return;
    }
    appState.lastMessage = parsed;
    appState.lastMessageReceivedAt = Date.now();
    appState.messageCount += 1;
    updateStatusUI();
    renderViewer(parsed);
  };

  ws.onerror = () => {
    appState.connectionStatus = "error";
    updateStatusUI();
    console.error("WebSocket error");
  };

  ws.onclose = () => {
    appState.connectionStatus = "disconnected";
    updateStatusUI();
    console.log("WebSocket closed");
  };
}

function playbackTick() {
  const frames = playbackState.frames;
  if (!frames || frames.length === 0) return;

  const i = playbackState.frameIndex;
  const frame = frames[i];
  playbackState.frameLabel = `${i + 1} / ${playbackState.totalFrames}`;

  const dets = frameToDetections(frame);
  appState.lastMessage = dets;
  appState.lastMessageReceivedAt = Date.now();
  appState.messageCount += 1;
  updateStatusUI();
  renderViewer(dets);

  playbackState.frameIndex = (i + 1) % playbackState.totalFrames;
}

/**
 * Option A: load CAM_FRONT_sample.json and advance frames on a timer (no WebSocket).
 */
async function startSamplePlayback() {
  appState.connectionStatus = "connecting";
  appState.messageCount = 0;
  updateStatusUI();

  try {
    const res = await fetch(SAMPLE_JSON_PATH);
    if (!res.ok) {
      throw new Error("HTTP " + res.status + " " + res.statusText);
    }
    const frames = await res.json();
    if (!Array.isArray(frames) || frames.length === 0) {
      throw new Error("Expected non-empty JSON array of frames");
    }

    playbackState.frames = frames;
    playbackState.totalFrames = frames.length;
    playbackState.frameIndex = 0;

    appState.connectionStatus = "playback";
    appState.messageCount = 0;

    playbackTick();
    playbackState.timerId = window.setInterval(playbackTick, PLAYBACK_INTERVAL_MS);
  } catch (err) {
    console.error("Sample playback:", err);
    appState.connectionStatus = "error";
    updateStatusUI();
  }
}

function setupDetectionToggle() {
  const toggle = document.getElementById("toggle-detections");
  if (!toggle) return;
  toggle.checked = appState.showDetections;
  toggle.addEventListener("change", () => {
    appState.showDetections = toggle.checked;
    if (appState.lastMessage) {
      renderViewer(appState.lastMessage);
    }
  });
}

setupDetectionToggle();

if (IS_PLAYBACK_MODE) {
  startSamplePlayback();
} else {
  connectWebSocket();
}
