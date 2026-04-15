// worker.js — runs off the main thread
// Handles: JSON.parse, base64→bytes, createImageBitmap
// Posts back a transferred ImageBitmap + detections so the main thread only renders.

const CLASS_NAMES = [
  "car", "truck", "trailer", "bus", "construction_vehicle",
  "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
];

let frameCount = 0;

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
      bounding_box: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
      track_id: trackId,
      confidence: conf
    });
  }
  return out;
}

self.onmessage = async (e) => {
  const frameId = ++frameCount;
  const { raw, sentAt } = e.data;        // sentAt = Date.now() from main thread
  const t0 = performance.now();          // worker received

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    self.postMessage({ type: "error", message: "JSON parse failed: " + err.message });
    return;
  }
  const t1 = performance.now();          // after JSON.parse

  if (!parsed || typeof parsed !== "object") {
    self.postMessage({ type: "error", message: "unexpected message type: " + typeof parsed });
    return;
  }

  if (parsed.tracks && parsed.metadata) {
    const meta = parsed.metadata;
    const cameraId = cameraIdFromMetadata(meta.camera);
    const detections = frameToDetections(parsed);
    const t2 = performance.now();        // after frameToDetections

    let bitmap = null;
    if (parsed.image) {
      try {
        const b64 = parsed.image;
        const binary = atob(b64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const t2b = performance.now();   // after atob + typed array
        const blob = new Blob([bytes], { type: "image/jpeg" });
        bitmap = await createImageBitmap(blob);
        const t3 = performance.now();   // after createImageBitmap

        const msgKB = (raw.length / 1024).toFixed(0);
        // console.log(
        //   `[worker #${frameId}] msg=${msgKB}KB` +
        //   ` | parse=${(t1-t0).toFixed(1)}ms` +
        //   ` | detections=${(t2-t1).toFixed(1)}ms` +
        //   ` | b64+bytes=${(t2b-t2).toFixed(1)}ms` +
        //   ` | createImageBitmap=${(t3-t2b).toFixed(1)}ms` +
        //   ` | worker total=${(t3-t0).toFixed(1)}ms`
        // );

        self.postMessage(
          { type: "frame", frameId, cameraId, detections, meta, bitmap, sentAt, workerDoneAt: Date.now() },
          [bitmap]
        );
        return;
      } catch (err) {
        console.error("Worker: bitmap decode failed", err);
      }
    }

    self.postMessage(
      { type: "frame", frameId, cameraId, detections, meta, bitmap: null, sentAt, workerDoneAt: Date.now() },
      []
    );
  } else {
    self.postMessage({ type: "legacy", data: parsed });
  }
};
