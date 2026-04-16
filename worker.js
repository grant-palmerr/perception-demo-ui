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
  const cam = cameraIdFromMetadata(frame.cam_id || meta.camera);
  const tracks = frame && frame.tracks ? frame.tracks : {};
  const out = [];

  for (const [trackId, track] of Object.entries(tracks)) {
    let x1, y1, x2, y2, conf, classId, anomalyScore, isAnomaly;

    if (Array.isArray(track)) {
      // Legacy array format: [x1, y1, x2, y2, conf, classId, anomaly_score?, is_anomaly?]
      if (track.length < 6) continue;
      [x1, y1, x2, y2, conf, classId] = track;
      anomalyScore = track.length >= 7 ? Number(track[6]) : null;
      isAnomaly = track.length >= 8 ? !!track[7] : (anomalyScore != null && anomalyScore > 0.5);
    } else if (track && typeof track === "object") {
      // New object format: { bbox, score, class_id, anomaly_score }
      const bbox = track.bbox;
      if (!Array.isArray(bbox) || bbox.length < 4) continue;
      [x1, y1, x2, y2] = bbox;
      conf = Number(track.score ?? 0);
      classId = Math.floor(Number(track.class_id ?? 0));
      anomalyScore = track.anomaly_score != null ? Number(track.anomaly_score) : null;
      isAnomaly = anomalyScore != null && anomalyScore > 0.5;
    } else {
      continue;
    }

    const objectClass = CLASS_NAMES[classId] ?? `class_${classId}`;
    out.push({
      camera_id: cam,
      object_class: objectClass,
      bounding_box: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
      track_id: trackId,
      confidence: conf,
      anomaly_score: anomalyScore,
      is_anomaly: isAnomaly
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
    const cameraId = cameraIdFromMetadata(parsed.cam_id || meta.camera);
    const detections = frameToDetections(parsed);
    const t2 = performance.now();        // after frameToDetections

    // DEBUG — remove once working
    if (frameCount <= 3) {
      console.log(`[worker #${frameCount}] cam_id=${parsed.cam_id} → cameraId=${cameraId}`);
      console.log(`[worker #${frameCount}] tracks keys:`, Object.keys(parsed.tracks).slice(0, 5));
      const firstKey = Object.keys(parsed.tracks)[0];
      if (firstKey != null) console.log(`[worker #${frameCount}] tracks[${firstKey}]:`, parsed.tracks[firstKey]);
      console.log(`[worker #${frameCount}] detections.length=${detections.length}`);
    }

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
