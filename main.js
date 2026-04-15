// Perception demo UI — mostly WebSocket stuff + drawing boxes on the cameras

// change this if your tunnel / port is different
const DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8000/tracking";

function getWebSocketUrl() {
  return localStorage.getItem("wsUrl") || DEFAULT_WEBSOCKET_URL;
}

function getHttpBaseUrl() {
  return getWebSocketUrl()
    .replace(/^wss?:\/\//, "http://")
    .replace(/\/[^/]+$/, "");
}

// ?playback=1 loads this file; needs a real http server or fetch breaks
const SAMPLE_JSON_PATH = "CAM_FRONT_sample.json";
const PLAYBACK_INTERVAL_MS = 150;

function getPlaybackModeFromUrl() {
  const p = new URLSearchParams(window.location.search).get("playback");
  return p === "1" || p === "true" || p === "yes";
}

const IS_PLAYBACK_MODE = getPlaybackModeFromUrl();

// maps class ids to strings — pull from tracking_pipeline.py on the board when we know the real list
const CLASS_NAMES = [
  "car", "truck", "trailer", "bus", "construction_vehicle",
  "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
];

// used when you're in playback mode (no socket)
const playbackState = {
  frames: null,
  totalFrames: 0,
  frameIndex: 0,
  timerId: null,
  frameLabel: "—"
};

// placeholder pics for now — the old azure links from email 404'd lol
const CAMERA_IMAGES = {
  front: "https://placehold.co/640x480/2a2a32/b8b8c8?text=front",
  front_left: "https://placehold.co/640x480/2a2a32/b8b8c8?text=front_left",
  front_right: "https://placehold.co/640x480/2a2a32/b8b8c8?text=front_right",
  back: "https://placehold.co/640x480/2a2a32/b8b8c8?text=back",
  back_left: "https://placehold.co/640x480/2a2a32/b8b8c8?text=back_left",
  back_right: "https://placehold.co/640x480/2a2a32/b8b8c8?text=back_right"
};

// json boxes are in "full cam" pixels (nuScenes front = 1600x900). we squish them down to match whatever img we're showing.
// liveImageSizes is populated from metadata.image_width/height when the backend sends real frames.
const ANNOTATION_SIZE_BY_CAMERA = {
  front: { w: 1600, h: 900 }
};

// per-camera sizes received live from the backend (overrides the static table above)
const liveImageSizes = {};

// decoded ImageBitmaps per camera — populated async via createImageBitmap
const liveBitmaps = {};

// rAF throttle: one pending flag per camera so cameras don't block each other
const rafPendingByCamera = new Map(); // cameraId → boolean

// rolling-window FPS: per-camera timestamp windows, displayed as average across active cameras
const FPS_WINDOW = 30;
const fpsWindowsPerCamera = new Map(); // cameraId → timestamp[]
const fpsLabel = document.getElementById("fps-label");

function recordFrame(cameraId) {
  if (!cameraId) return;
  const now = Date.now();
  if (!fpsWindowsPerCamera.has(cameraId)) fpsWindowsPerCamera.set(cameraId, []);
  const ts = fpsWindowsPerCamera.get(cameraId);
  ts.push(now);
  if (ts.length > FPS_WINDOW) ts.shift();

  if (!fpsLabel) return;
  let total = 0;
  let count = 0;
  for (const [, stamps] of fpsWindowsPerCamera) {
    if (stamps.length < 2) continue;
    const spanSec = (stamps[stamps.length - 1] - stamps[0]) / 1000;
    total += (stamps.length - 1) / spanSec;
    count++;
  }
  if (count > 0) fpsLabel.textContent = (total / count).toFixed(1) + " fps";
}

function resetFps() {
  fpsWindowsPerCamera.clear();
  if (fpsLabel) fpsLabel.textContent = "— fps";
}

function scheduleRender(cameraId, frameId, workerDoneAt) {
  const handoffMs = workerDoneAt != null ? Date.now() - workerDoneAt : null;

  if (rafPendingByCamera.get(cameraId)) {
    if (frameId != null) console.log(`[main  #${frameId}] dropped — rAF already pending for ${cameraId} (handoff=${handoffMs}ms)`);
    return;
  }
  rafPendingByCamera.set(cameraId, true);
  requestAnimationFrame(() => {
    rafPendingByCamera.set(cameraId, false);
    rerenderCamera(cameraId);
  });
}

// Web Worker — handles JSON.parse, base64 decode, and createImageBitmap off the main thread
const frameWorker = new Worker("worker.js");

frameWorker.onmessage = (e) => {
  const msg = e.data;

  if (msg.type === "frame") {
    const { frameId, cameraId, detections, meta, bitmap, workerDoneAt } = msg;
    if (bitmap) {
      if (liveBitmaps[cameraId]) liveBitmaps[cameraId].close();
      liveBitmaps[cameraId] = bitmap;
    }
    if (meta.image_width && meta.image_height) {
      liveImageSizes[cameraId] = { w: meta.image_width, h: meta.image_height };
    }
    latestDetectionsByCamera.set(cameraId, detections);
    updateTrackTable(cameraId, detections);
    appState.lastMessage = detections;
    appState.lastMessageReceivedAt = Date.now();
    appState.messageCount += 1;
    recordFrame(cameraId);
    updateStatusUI();
    scheduleRender(cameraId, frameId, workerDoneAt);
  } else if (msg.type === "legacy") {
    appState.lastMessage = msg.data;
    appState.lastMessageReceivedAt = Date.now();
    appState.messageCount += 1;
    updateStatusUI();
    renderViewer(msg.data);
  } else if (msg.type === "error") {
    console.error("Worker:", msg.message);
  }
};

// how much to scale x and y so boxes line up with the image
function getAnnotationScale(cameraId, canvasW, canvasH) {
  const ref = liveImageSizes[cameraId] || ANNOTATION_SIZE_BY_CAMERA[cameraId];
  if (!ref || !ref.w || !ref.h || ref.w <= 0 || ref.h <= 0) {
    return { sx: 1, sy: 1 };
  }
  return { sx: canvasW / ref.w, sy: canvasH / ref.h };
}

// track_id → { detection, cameraId, disappearedAt, missedFrames }
// disappearedAt is null while the track is active; set to Date.now() when it vanishes.
// tracks linger in the table (struck through) for TRACK_LINGER_MS after disappearing.
const trackRegistry = new Map();
const TRACK_LINGER_MS = 5000;
// mirrors sort_max_age from pipeline config — updated when config loads or is applied
let configSortMaxAge = 5;

const trackTbody = document.getElementById("track-tbody");

function updateTrackTable(cameraId, detections) {
  const now = Date.now();
  const activeIds = new Set(detections.map((d) => String(d.track_id)));

  // upsert active tracks — reset missed-frame counter on reappearance
  for (const d of detections) {
    trackRegistry.set(String(d.track_id), { detection: d, cameraId, disappearedAt: null, missedFrames: 0 });
  }

  // increment missed-frame counter for absent tracks; only mark inactive after GRACE_FRAMES
  for (const [key, entry] of trackRegistry) {
    if (entry.cameraId === cameraId && !activeIds.has(key) && entry.disappearedAt === null) {
      entry.missedFrames += 1;
      if (entry.missedFrames >= configSortMaxAge) {
        entry.disappearedAt = now;
      }
    }
  }

  renderTrackTable();
}

// tracks whose backend has stopped reporting them but haven't expired yet (missedFrames in [1, configSortMaxAge))
function getGracePeriodDetections(cameraId) {
  const result = [];
  for (const [, entry] of trackRegistry) {
    if (entry.cameraId === cameraId && entry.disappearedAt === null && entry.missedFrames > 0) {
      result.push(entry.detection);
    }
  }
  return result;
}

function rerenderAllCameras() {
  for (const cameraId of Object.keys(liveBitmaps)) rerenderCamera(cameraId);
}

// redraw a single camera slot using its stored detections — used for hover highlight
function rerenderCamera(cameraId) {
  const slot = document.querySelector(`.camera-slot[data-camera="${cameraId}"]`);
  if (!slot) return;
  const canvas = slot.querySelector("canvas");
  if (!canvas) return;
  const bitmap = liveBitmaps[cameraId];
  if (!bitmap) return;

  showCameraSlot(cameraId);
  const img = slot.querySelector("img");
  if (img) img.hidden = true;
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }

  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);

  const cssScale = canvas.width / canvas.clientWidth;
  ctx.lineWidth = Math.round(cssScale);

  const detections = (latestDetectionsByCamera.get(cameraId) || [])
    .filter(d => !appState.hiddenClasses.has(d.object_class));
  for (const d of detections) drawBoundingBox(ctx, d, 1, 1);

  const grace = getGracePeriodDetections(cameraId)
    .filter(d => !appState.hiddenClasses.has(d.object_class));
  if (grace.length > 0) {
    ctx.globalAlpha = 0.45;
    for (const d of grace) drawBoundingBox(ctx, d, 1, 1);
    ctx.globalAlpha = 1;
  }
}

function renderTrackTable() {
  if (!trackTbody) return;

  const all = [...trackRegistry.values()];
  all.sort((a, b) => Number(b.detection.track_id) - Number(a.detection.track_id));

  trackTbody.innerHTML = "";

  if (all.length === 0) {
    trackTbody.innerHTML = '<tr><td colspan="2" class="track-empty">No active tracks.</td></tr>';
    return;
  }

  for (const { detection: d, disappearedAt } of all) {
    if (appState.hiddenClasses.has(d.object_class)) continue;
    const tr = document.createElement("tr");
    if (disappearedAt !== null) tr.classList.add("track-row-inactive");

    const tdId = document.createElement("td");
    tdId.textContent = "#" + d.track_id;

    const tdClass = document.createElement("td");
    tdClass.textContent = (d.object_class || "unknown").replace(/_/g, " ");
    if (disappearedAt === null) tdClass.style.color = classColor(d.object_class);

    if (String(d.track_id) === String(appState.lockedTrackId)) {
      tr.classList.add("track-row-locked");
    }

    tr.addEventListener("mouseenter", () => {
      appState.hoveredTrackId = d.track_id;
      rerenderAllCameras();
    });

    tr.addEventListener("mouseleave", () => {
      appState.hoveredTrackId = null;
      rerenderAllCameras();
    });

    tr.addEventListener("click", () => {
      const prev = appState.lockedTrackId;
      appState.lockedTrackId = prev === d.track_id ? null : d.track_id;
      renderTrackTable();
      rerenderAllCameras();
    });

    tr.appendChild(tdId);
    tr.appendChild(tdClass);
    trackTbody.appendChild(tr);
  }
}

// purge tracks that have lingered past TRACK_LINGER_MS
setInterval(() => {
  const now = Date.now();
  let changed = false;
  for (const [key, entry] of trackRegistry) {
    if (entry.disappearedAt !== null && now - entry.disappearedAt >= TRACK_LINGER_MS) {
      trackRegistry.delete(key);
      changed = true;
    }
  }
  if (changed) renderTrackTable();
}, 1000);

const appState = {
  connectionStatus: "disconnected",
  lastMessage: null,
  lastMessageReceivedAt: null,
  messageCount: 0,
  hiddenClasses: new Set(),
  hoveredTrackId: null,
  lockedTrackId: null,
  sequenceIdx: null,
  sequenceState: "idle"  // "idle" | "playing" | "ended" | "done"
};

// latest detections per camera — used to redraw a single camera slot on hover change
const latestDetectionsByCamera = new Map();

const wsDot = document.getElementById("ws-status-dot");
const wsLabel = document.getElementById("ws-status-label");

// websocket can send an array or { detections: [...] } — normalize it
function getDetections(msg) {
  if (!msg) return [];
  if (Array.isArray(msg) && msg.length > 0 && typeof msg[0] === "object" && "bounding_box" in (msg[0] ?? {})) {
    return msg;
  }
  if (Array.isArray(msg)) return msg;
  return msg.detections ?? msg.objects ?? msg.boxes ?? [];
}

// CAM_FRONT -> front, etc. so it matches the html data-camera attrs
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

// turns one json frame into the shape renderViewer already likes
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

const CLASS_COLORS = {
  car:                   "#00aaff",
  truck:                 "#ff8800",
  trailer:               "#ff6600",
  bus:                   "#ff4400",
  construction_vehicle:  "#ffcc00",
  bicycle:               "#00ffcc",
  motorcycle:            "#00ffaa",
  pedestrian:            "#ff4dff",
  traffic_cone:          "#ff2222",
  barrier:               "#aaaaaa"
};

function classColor(objectClass) {
  return CLASS_COLORS[objectClass] ?? "#ffffff";
}

// draw one box; sx/sy shrink coords from annotation space to canvas
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
  const color = classColor(d.object_class);
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
  const highlightId = appState.hoveredTrackId ?? appState.lockedTrackId;
  if (highlightId != null && String(d.track_id) === String(highlightId)) {
    ctx.fillStyle = color + "33"; // ~20% opacity fill
    ctx.fill();
  }
  ctx.stroke();
  if (d.object_class) {
    ctx.fillStyle = color;
    const fontPx = Math.max(10, Math.round(12 * Math.min(scaleX, scaleY)));
    ctx.font = fontPx + "px sans-serif";
    ctx.fillText(d.object_class, points[0][0], points[0][1] - 4);
  }
}

function updateGridLayout() {
  const viewer = document.getElementById("viewer");
  if (!viewer) return;
  const n = viewer.querySelectorAll(".camera-slot.active").length;
  // pick column count that fills the space sensibly for 1–6 cameras
  const cols = [0, 1, 2, 3, 2, 3, 3][n] ?? 3;
  viewer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
}

function showCameraSlot(cameraId) {
  const slot = document.querySelector(`.camera-slot[data-camera="${cameraId}"]`);
  if (!slot) return;
  slot.classList.add("active");
  updateGridLayout();
}

// slap detections on every camera tile (most will be empty unless we have data for them)
function renderViewer(message) {
  if (!message) return;
  const detections = getDetections(message);
  const slots = document.querySelectorAll(".camera-slot");

  slots.forEach((slot) => {
    const cameraId = slot.dataset.camera;
    const img = slot.querySelector("img");
    const canvas = slot.querySelector("canvas");
    if (!canvas) return;

    const cameraDetections = detections.filter(
      (d) => (d.camera_id || "").toLowerCase().replace(/\s/g, "_") === cameraId
        && !appState.hiddenClasses.has(d.object_class)
    );

    const bitmap = liveBitmaps[cameraId];

    if (bitmap) {
      // fast path: image already decoded — draw image + boxes on one canvas
      showCameraSlot(cameraId);
      if (img) img.hidden = true;
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      const ctx = canvas.getContext("2d");
      ctx.drawImage(bitmap, 0, 0);
      // canvas is CSS-scaled down — multiply lineWidth by the inverse scale so
      // it always appears as 1 screen pixel regardless of display size
      const cssScale = canvas.width / canvas.clientWidth;
      ctx.lineWidth = Math.round(cssScale);
      for (const d of cameraDetections) drawBoundingBox(ctx, d, 1, 1);
      const grace = getGracePeriodDetections(cameraId)
        .filter(d => !appState.hiddenClasses.has(d.object_class));
      if (grace.length > 0) {
        ctx.globalAlpha = 0.45;
        for (const d of grace) drawBoundingBox(ctx, d, 1, 1);
        ctx.globalAlpha = 1;
      }
      return;
    }

    // fallback: placeholder img + canvas overlay (playback / no live image yet)
    if (!img || !CAMERA_IMAGES[cameraId]) return;

    const drawBoxes = () => {
      if (!img.naturalWidth) return;
      if (cameraDetections.length > 0) showCameraSlot(cameraId);
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      // draw image onto canvas (same approach as the live bitmap path)
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      if (cameraDetections.length > 0) {
        const { sx, sy } = getAnnotationScale(cameraId, canvas.width, canvas.height);
        ctx.lineWidth = Math.max(1, Math.min(2, 2 * Math.min(sx, sy)));
        for (const d of cameraDetections) drawBoundingBox(ctx, d, sx, sy);
      }
    };

    const targetUrl = CAMERA_IMAGES[cameraId];
    if (img.src !== targetUrl) {
      img.src = targetUrl;
      img.onload = drawBoxes;
    } else if (img.complete) {
      drawBoxes();
    }
  });
}

function updateStatusUI() {
  if (!wsDot || !wsLabel) return;

  const status = IS_PLAYBACK_MODE
    ? (appState.connectionStatus === "playback" ? "playback" : appState.connectionStatus)
    : appState.connectionStatus;

  wsDot.dataset.status = status;

  if (IS_PLAYBACK_MODE) {
    const labels = {
      connecting: "loading…",
      playback: `playback — frame ${playbackState.frameLabel}`,
      error: "playback failed"
    };
    wsLabel.textContent = labels[appState.connectionStatus] ?? appState.connectionStatus;
  } else {
    const labels = {
      disconnected: "disconnected",
      connecting: "connecting…",
      connected: "connected",
      error: "connection error",
      "playback-done": "playback complete"
    };
    wsLabel.textContent = labels[appState.connectionStatus] ?? appState.connectionStatus;
  }
}

let activeSocket = null;

// ── Health polling ────────────────────────────────────────────────────────────

let healthPollTimer = null;

const healthDot = document.getElementById("health-inference-dot");
const healthLabel = document.getElementById("health-label");
const healthSubsLabel = document.getElementById("health-subs-label");

async function pollHealth() {
  try {
    const res = await fetch(getHttpBaseUrl() + "/health");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    if (healthDot) healthDot.dataset.status = data.inference_running ? "running" : "stopped";
    if (healthLabel) healthLabel.textContent = data.inference_running ? "inference on" : "inference off";
    if (healthSubsLabel) {
      const n = data.subscribers ?? 0;
      healthSubsLabel.textContent = n > 0 ? `${n} sub${n === 1 ? "" : "s"}` : "";
    }
  } catch (_) {
    if (healthDot) healthDot.dataset.status = "";
    if (healthLabel) healthLabel.textContent = "—";
    if (healthSubsLabel) healthSubsLabel.textContent = "";
  }
}

function startHealthPolling() {
  if (IS_PLAYBACK_MODE) return;
  stopHealthPolling();
  pollHealth();
  healthPollTimer = setInterval(pollHealth, 5000);
}

function stopHealthPolling() {
  if (healthPollTimer != null) {
    clearInterval(healthPollTimer);
    healthPollTimer = null;
  }
  if (healthDot) healthDot.dataset.status = "";
  if (healthLabel) healthLabel.textContent = "—";
  if (healthSubsLabel) healthSubsLabel.textContent = "";
  resetFps();
}

// ── Sequence banner ───────────────────────────────────────────────────────────

const seqBanner = document.getElementById("seq-banner");
const seqBannerText = document.getElementById("seq-banner-text");
let seqBannerAutoHideTimer = null;

function showSeqBanner(text, state) {
  if (!seqBanner || !seqBannerText) return;
  // remove any reconnect button from a previous "done" state
  const existing = seqBanner.querySelector(".seq-reconnect-btn");
  if (existing) existing.remove();
  seqBannerText.textContent = text;
  seqBanner.dataset.state = state;
  seqBanner.hidden = false;
}

function hideSeqBanner() {
  if (seqBanner) seqBanner.hidden = true;
}

function handleLifecycleSignal(msg) {
  if (msg.status === "sequence_start") {
    appState.sequenceIdx = msg.seq_idx;
    appState.sequenceState = "playing";
    showSeqBanner(`Sequence ${msg.seq_idx} started`, "start");
    clearTimeout(seqBannerAutoHideTimer);
    seqBannerAutoHideTimer = setTimeout(() => {
      if (appState.sequenceState === "playing") hideSeqBanner();
    }, 3000);

  } else if (msg.status === "sequence_end") {
    appState.sequenceState = "ended";
    clearTimeout(seqBannerAutoHideTimer);
    showSeqBanner(`Sequence ${msg.seq_idx} ended`, "end");

  } else if (msg.status === "done") {
    appState.sequenceState = "done";
    appState.connectionStatus = "playback-done";
    wsDot.dataset.status = "playback";
    updateStatusUI();
    clearTimeout(seqBannerAutoHideTimer);
    showSeqBanner("Playback complete — reconnect to replay", "done");
    const reconnectBtn = document.createElement("button");
    reconnectBtn.className = "seq-reconnect-btn";
    reconnectBtn.textContent = "Reconnect";
    reconnectBtn.addEventListener("click", () => {
      hideSeqBanner();
      connectWebSocket();
    });
    if (seqBanner) seqBanner.appendChild(reconnectBtn);
  }
}

// ── Camera grid sync ──────────────────────────────────────────────────────────

function handleCameraListChange(cameras) {
  const activeIds = new Set((cameras || []).map(cameraIdFromMetadata));
  document.querySelectorAll(".camera-slot").forEach(slot => {
    slot.classList.toggle("active", activeIds.has(slot.dataset.camera));
  });
  updateGridLayout();
}

// ── WebSocket connection ──────────────────────────────────────────────────────

function disconnectWebSocket() {
  if (!activeSocket) return;
  try {
    if (activeSocket.readyState === WebSocket.OPEN) {
      activeSocket.send(JSON.stringify({ command: "stop" }));
    }
  } catch (_) {}
  activeSocket.onclose = null;
  activeSocket.close();
  activeSocket = null;
  stopHealthPolling();
  appState.connectionStatus = "disconnected";
  updateStatusUI();
}

window.addEventListener("beforeunload", disconnectWebSocket);

function connectWebSocket() {
  disconnectWebSocket();

  const url = getWebSocketUrl();
  appState.connectionStatus = "connecting";
  appState.messageCount = 0;
  appState.sequenceState = "idle";
  hideSeqBanner();
  updateStatusUI();

  const ws = new WebSocket(url);
  activeSocket = ws;

  ws.onopen = () => {
    appState.connectionStatus = "connected";
    updateStatusUI();
    startHealthPolling();
    console.log("WebSocket connected to", url);
  };

  ws.onmessage = (event) => {
    // sniff short messages for lifecycle signals before forwarding to worker
    if (event.data.length < 200) {
      let peeked;
      try { peeked = JSON.parse(event.data); } catch (_) { peeked = null; }
      if (peeked && typeof peeked.status === "string") {
        handleLifecycleSignal(peeked);
        return;
      }
    }
    // hand the raw string to the worker — JSON.parse + image decode happen off-thread
    frameWorker.postMessage({ raw: event.data, sentAt: Date.now() });
  };

  ws.onerror = () => {
    appState.connectionStatus = "error";
    updateStatusUI();
    console.error("WebSocket error");
  };

  ws.onclose = () => {
    stopHealthPolling();
    if (appState.connectionStatus !== "playback-done") {
      appState.connectionStatus = "disconnected";
      updateStatusUI();
    }
    console.log("WebSocket closed");
  };
}

// next frame in playback mode
function playbackTick() {
  const frames = playbackState.frames;
  if (!frames || frames.length === 0) return;

  const i = playbackState.frameIndex;
  const frame = frames[i];
  playbackState.frameLabel = `${i + 1} / ${playbackState.totalFrames}`;

  const dets = frameToDetections(frame);
  const cameraId = dets.length > 0 ? dets[0].camera_id : null;
  if (cameraId) {
    latestDetectionsByCamera.set(cameraId, dets);
    updateTrackTable(cameraId, dets);
  }
  appState.lastMessage = dets;
  appState.lastMessageReceivedAt = Date.now();
  appState.messageCount += 1;
  updateStatusUI();
  renderViewer(dets);

  playbackState.frameIndex = (i + 1) % playbackState.totalFrames;
}

// fake a "live" stream by reading the sample file and stepping frames on a timer
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

// "All" + per-class visibility checkboxes
function setupClassFilters() {
  const container = document.getElementById("class-filters");
  if (!container) return;

  const classCbs = new Map(); // name → checkbox element

  function rerender() {
    if (appState.lastMessage) renderViewer(appState.lastMessage);
    renderTrackTable();
  }

  // "All" checkbox
  const allLabel = document.createElement("label");
  allLabel.className = "class-filter-label class-filter-all";

  const allCb = document.createElement("input");
  allCb.type = "checkbox";
  allCb.checked = true;
  allCb.addEventListener("change", () => {
    classCbs.forEach((cb, name) => {
      cb.checked = allCb.checked;
      if (allCb.checked) appState.hiddenClasses.delete(name);
      else appState.hiddenClasses.add(name);
    });
    rerender();
  });

  allLabel.appendChild(allCb);
  allLabel.appendChild(document.createTextNode("All"));
  container.appendChild(allLabel);

  // divider
  const sep = document.createElement("span");
  sep.className = "class-filter-sep";
  container.appendChild(sep);

  // per-class checkboxes
  for (const name of CLASS_NAMES) {
    const color = classColor(name);

    const label = document.createElement("label");
    label.className = "class-filter-label";
    label.style.color = color;

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.style.accentColor = color;
    cb.addEventListener("change", () => {
      if (cb.checked) {
        appState.hiddenClasses.delete(name);
      } else {
        appState.hiddenClasses.add(name);
        allCb.checked = false; // visually uncheck All when any class is deselected
      }
      rerender();
    });

    classCbs.set(name, cb);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(name.replace(/_/g, " ")));
    container.appendChild(label);
  }
}

setupClassFilters();

// populate ws url input and wire up the connect button
function setupWsUrlInput() {
  const input = document.getElementById("ws-url-input");
  const btn = document.getElementById("ws-connect-btn");
  const group = document.getElementById("ws-url-group");
  if (!input || !btn) return;

  // hide the control in playback mode — it's irrelevant there
  if (IS_PLAYBACK_MODE && group) {
    group.style.display = "none";
    return;
  }

  input.value = getWebSocketUrl();

  btn.addEventListener("click", () => {
    const url = input.value.trim();
    if (!url) return;
    localStorage.setItem("wsUrl", url);
    connectWebSocket();
  });

  // also connect on Enter
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") btn.click();
  });
}

setupWsUrlInput();

// clicking a canvas selects the smallest bounding box under the cursor
function setupCanvasClickHandlers() {
  document.querySelectorAll(".camera-slot").forEach(slot => {
    const cameraId = slot.dataset.camera;
    const canvas = slot.querySelector("canvas");
    if (!canvas) return;

    canvas.addEventListener("click", (e) => {
      if (!canvas.width || !canvas.height) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top) * scaleY;

      const detections = (latestDetectionsByCamera.get(cameraId) || [])
        .filter(d => !appState.hiddenClasses.has(d.object_class));

      // pick the smallest box that contains the click (handles overlapping boxes)
      let best = null;
      let bestArea = Infinity;
      for (const d of detections) {
        const bbox = d.bounding_box ?? d.bbox ?? d.box;
        if (!Array.isArray(bbox) || bbox.length < 4) continue;
        const xs = bbox.map(p => p[0]);
        const ys = bbox.map(p => p[1]);
        const x1 = Math.min(...xs), x2 = Math.max(...xs);
        const y1 = Math.min(...ys), y2 = Math.max(...ys);
        if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) {
          const area = (x2 - x1) * (y2 - y1);
          if (area < bestArea) { bestArea = area; best = d; }
        }
      }

      // toggle: clicking the already-locked box unlocks; clicking empty space also unlocks
      const prev = appState.lockedTrackId;
      appState.lockedTrackId = (best && String(best.track_id) !== String(prev)) ? best.track_id : null;
      renderTrackTable();
      rerenderAllCameras();

      // scroll the locked row into view in the side table
      if (appState.lockedTrackId != null) {
        const lockedRow = trackTbody.querySelector(".track-row-locked");
        if (lockedRow) lockedRow.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });
  });
}

setupCanvasClickHandlers();

// ── Seq banner dismiss ────────────────────────────────────────────────────────

const seqBannerDismiss = document.getElementById("seq-banner-dismiss");
if (seqBannerDismiss) seqBannerDismiss.addEventListener("click", hideSeqBanner);

// ── Config panel ──────────────────────────────────────────────────────────────

const ALL_CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"];

let configDraft = {};
let configFull = null; // last full config from server (includes read-only fields)

function openConfigPanel() {
  const overlay = document.getElementById("config-overlay");
  const panel = document.getElementById("config-panel");
  const body = document.getElementById("config-panel-body");
  const applyBtn = document.getElementById("config-apply-btn");
  if (!overlay || !panel) return;
  overlay.hidden = false;
  panel.hidden = false;
  if (body) body.innerHTML = '<p class="config-loading">Loading\u2026</p>';
  if (applyBtn) applyBtn.disabled = true;
  loadConfig();
}

function closeConfigPanel() {
  const overlay = document.getElementById("config-overlay");
  const panel = document.getElementById("config-panel");
  const feedback = document.getElementById("config-feedback");
  if (overlay) overlay.hidden = true;
  if (panel) panel.hidden = true;
  if (feedback) { feedback.textContent = ""; feedback.dataset.state = ""; }
}

async function loadConfig() {
  const body = document.getElementById("config-panel-body");
  const applyBtn = document.getElementById("config-apply-btn");
  try {
    const res = await fetch(getHttpBaseUrl() + "/config");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const cfg = await res.json();
    configFull = cfg;
    configSortMaxAge = cfg.sort_max_age ?? configSortMaxAge;
    configDraft = {
      threshold: cfg.threshold,
      cameras: [...(cfg.cameras || [])],
      loop_sequences: cfg.loop_sequences,
      sort_max_age: cfg.sort_max_age,
      sort_min_hits: cfg.sort_min_hits,
      sort_iou_threshold: cfg.sort_iou_threshold
    };
    renderConfigForm(cfg);
    if (applyBtn) applyBtn.disabled = false;
  } catch (err) {
    if (body) body.innerHTML = `<p class="config-loading" style="color:#e74c3c">Failed to load config: ${err.message}</p>`;
  }
}

function renderConfigForm(cfg) {
  const body = document.getElementById("config-panel-body");
  if (!body) return;
  body.innerHTML = "";

  // threshold slider
  const threshField = document.createElement("div");
  threshField.className = "config-field";
  const threshLabel = document.createElement("span");
  threshLabel.className = "config-field-label";
  threshLabel.textContent = "Confidence threshold";
  const threshRow = document.createElement("div");
  threshRow.className = "config-range-row";
  const threshSlider = document.createElement("input");
  threshSlider.type = "range";
  threshSlider.min = "0"; threshSlider.max = "1"; threshSlider.step = "0.01";
  threshSlider.value = String(cfg.threshold);
  const threshVal = document.createElement("span");
  threshVal.className = "config-range-value";
  threshVal.textContent = Number(cfg.threshold).toFixed(2);
  threshSlider.addEventListener("input", () => {
    threshVal.textContent = Number(threshSlider.value).toFixed(2);
    configDraft.threshold = Number(threshSlider.value);
  });
  threshRow.appendChild(threshSlider);
  threshRow.appendChild(threshVal);
  threshField.appendChild(threshLabel);
  threshField.appendChild(threshRow);
  body.appendChild(threshField);

  // cameras chip select
  const camField = document.createElement("div");
  camField.className = "config-field";
  const camLabel = document.createElement("span");
  camLabel.className = "config-field-label";
  camLabel.textContent = "Active cameras";
  const camGrid = document.createElement("div");
  camGrid.className = "config-cameras-grid";
  for (const cam of ALL_CAMERAS) {
    const chip = document.createElement("span");
    chip.className = "config-cam-chip" + (configDraft.cameras.includes(cam) ? " selected" : "");
    chip.textContent = cam.replace(/^CAM_/, "").replace(/_/g, " ").toLowerCase();
    chip.addEventListener("click", () => {
      const idx = configDraft.cameras.indexOf(cam);
      if (idx >= 0) configDraft.cameras.splice(idx, 1);
      else configDraft.cameras.push(cam);
      chip.classList.toggle("selected", configDraft.cameras.includes(cam));
    });
    camGrid.appendChild(chip);
  }
  const camHint = document.createElement("span");
  camHint.className = "config-field-hint";
  camHint.textContent = "Click to toggle cameras";
  camField.appendChild(camLabel);
  camField.appendChild(camGrid);
  camField.appendChild(camHint);
  body.appendChild(camField);

  // loop_sequences toggle
  const loopField = document.createElement("div");
  loopField.className = "config-field";
  const loopLabel = document.createElement("label");
  loopLabel.className = "config-toggle";
  const loopCb = document.createElement("input");
  loopCb.type = "checkbox";
  loopCb.checked = !!cfg.loop_sequences;
  loopCb.addEventListener("change", () => { configDraft.loop_sequences = loopCb.checked; });
  loopLabel.appendChild(loopCb);
  loopLabel.appendChild(document.createTextNode("Loop sequences"));
  loopField.appendChild(loopLabel);
  body.appendChild(loopField);

  // numeric fields: sort params
  const numericFields = [
    { key: "sort_max_age", label: "SORT max age", min: 1, hint: "Frames to keep a missing track alive" },
    { key: "sort_min_hits", label: "SORT min hits", min: 1, hint: "Frames before confirming a new track" },
    { key: "sort_iou_threshold", label: "SORT IoU threshold", min: 0, max: 1, step: 0.01, hint: "Minimum IoU to associate a detection" }
  ];
  for (const { key, label, min, max, step, hint } of numericFields) {
    const field = document.createElement("div");
    field.className = "config-field";
    const lbl = document.createElement("span");
    lbl.className = "config-field-label";
    lbl.textContent = label;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.value = String(cfg[key]);
    if (min != null) inp.min = String(min);
    if (max != null) inp.max = String(max);
    if (step != null) inp.step = String(step);
    inp.addEventListener("input", () => { configDraft[key] = Number(inp.value); });
    const hintEl = document.createElement("span");
    hintEl.className = "config-field-hint";
    hintEl.textContent = hint;
    field.appendChild(lbl);
    field.appendChild(inp);
    field.appendChild(hintEl);
    body.appendChild(field);
  }
}

async function applyConfig() {
  const applyBtn = document.getElementById("config-apply-btn");
  const feedback = document.getElementById("config-feedback");
  if (applyBtn) applyBtn.disabled = true;
  if (feedback) { feedback.textContent = ""; feedback.dataset.state = ""; }

  const merged = Object.assign({}, configFull, configDraft);
  try {
    const res = await fetch(getHttpBaseUrl() + "/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(merged)
    });
    if (!res.ok) throw new Error("HTTP " + res.status + " " + res.statusText);
    const newCfg = await res.json();
    configFull = newCfg;
    configSortMaxAge = newCfg.sort_max_age ?? configSortMaxAge;
    if (feedback) { feedback.textContent = "Applied"; feedback.dataset.state = "ok"; }
    setTimeout(() => { if (feedback) { feedback.textContent = ""; feedback.dataset.state = ""; } }, 3000);
    handleCameraListChange(newCfg.cameras);
  } catch (err) {
    if (feedback) { feedback.textContent = "Error: " + err.message; feedback.dataset.state = "err"; }
  } finally {
    if (applyBtn) applyBtn.disabled = false;
  }
}

function setupConfigPanel() {
  const openBtn = document.getElementById("config-open-btn");
  const closeBtn = document.getElementById("config-close-btn");
  const overlay = document.getElementById("config-overlay");
  const applyBtn = document.getElementById("config-apply-btn");

  if (openBtn) openBtn.addEventListener("click", openConfigPanel);
  if (closeBtn) closeBtn.addEventListener("click", closeConfigPanel);
  if (overlay) overlay.addEventListener("click", closeConfigPanel);
  if (applyBtn) applyBtn.addEventListener("click", applyConfig);

  document.addEventListener("keydown", (e) => {
    const panel = document.getElementById("config-panel");
    if (e.key === "Escape" && panel && !panel.hidden) closeConfigPanel();
  });
}

setupConfigPanel();

// ── Startup ───────────────────────────────────────────────────────────────────

// url decides which path we use
if (IS_PLAYBACK_MODE) {
  startSamplePlayback();
} else {
  // silently load config to initialize the camera grid before first live frame
  fetch(getHttpBaseUrl() + "/config")
    .then(r => r.ok ? r.json() : null)
    .then(cfg => {
      if (!cfg) return;
      if (cfg.cameras) handleCameraListChange(cfg.cameras);
      if (cfg.sort_max_age) configSortMaxAge = cfg.sort_max_age;
    })
    .catch(() => {});
  connectWebSocket();
}
