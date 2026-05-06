import onnxruntime as rt
import numpy as np
import cv2
import logging
import threading
import time

from collections import deque
from dataclasses import dataclass, field
from trackers import FastSort
from utils import create_inference_session, associate_labels_to_tracks

logger = logging.getLogger('AnomalyTrackingPipeline')

CLASS_NAMES = [
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
]


@dataclass
class AnomalyCameraState:
    """Per-camera state: tracker + feature history + anomaly scores."""
    tracker: FastSort
    persistent_tracks: dict = field(default_factory=dict)
    # track_id -> deque of feature vectors, each shape (9,)
    track_histories: dict = field(default_factory=lambda: {})
    # track_id -> (cx, cy, bw, bh) from the previous frame for computing deltas
    prev_bbox_features: dict = field(default_factory=lambda: {})
    # track_id -> latest anomaly score
    anomaly_scores: dict = field(default_factory=lambda: {})

    @classmethod
    def from_args(cls, args) -> "AnomalyCameraState":
        return cls(
            tracker=FastSort(
                max_age=args.sort_max_age,
                min_hits=args.sort_min_hits,
                iou_threshold=args.sort_iou_threshold,
            )
        )


@dataclass
class FrameTimings:
    """Timing breakdown for a single frame."""
    detection_ms: float = 0.0
    filtering_ms: float = 0.0
    tracking_ms: float = 0.0
    feature_extraction_ms: float = 0.0
    anomaly_scoring_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "detection_ms": round(self.detection_ms, 3),
            "filtering_ms": round(self.filtering_ms, 3),
            "tracking_ms": round(self.tracking_ms, 3),
            "feature_extraction_ms": round(self.feature_extraction_ms, 3),
            "anomaly_scoring_ms": round(self.anomaly_scoring_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


class AnomalyTrackingPipeline:
    
    _SORT_PARAMS = ("max_age", "min_hits", "iou_threshold")
    
    def __init__(self, args, camera_ids: list[str] | None = None):
        self.args = args
        self._inference_lock = threading.Lock()
        self._config_lock = threading.RLock()

        self.window_size = getattr(args, 'predictor_window_size', 16)
        self.anomaly_threshold = getattr(args, 'anomaly_threshold', 0.5)
        self.anomaly_scale = getattr(args, 'anomaly_scale', 0.01)

        self._init_detector()
        self._init_predictor()

        ids = camera_ids if camera_ids is not None else [0]
        self.cameras: dict[str | int, AnomalyCameraState] = {
            cam_id: AnomalyCameraState.from_args(args) for cam_id in ids
        }

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def add_camera(self, cam_id: str | int) -> None:
        if cam_id in self.cameras:
            raise ValueError(f"Camera '{cam_id}' already registered.")
        self.cameras[cam_id] = AnomalyCameraState.from_args(self.args)

    def remove_camera(self, cam_id: str | int) -> None:
        self.cameras.pop(cam_id, None)

    def reset_camera(self, cam_id: str | int) -> None:
        if cam_id not in self.cameras:
            raise KeyError(f"Unknown camera '{cam_id}'.")
        self.cameras[cam_id] = AnomalyCameraState.from_args(self.args)

    def reset_all(self) -> None:
        for cam_id in list(self.cameras):
            self.cameras[cam_id] = AnomalyCameraState.from_args(self.args)

    # ------------------------------------------------------------------
    # Detector (TIDL-accelerated)
    # ------------------------------------------------------------------

    def _init_detector(self) -> None:
        logger.info("Creating YOLOX inference session...")
        self.det_session = create_inference_session(
            self.args.model_path,
            self.args.prototxt_path,
            self.args.artifacts_folder,
        )
        self.input_details = self.det_session.get_inputs()
        self.input_name   = self.input_details[0].name
        self.input_shape  = self.input_details[0].shape
        self.input_dtype  = (
            np.uint8
            if self.input_details[0].type == "tensor(uint8)"
            else np.float32
        )
        self.h_in, self.w_in = self.input_shape[2], self.input_shape[3]

        warmup = np.zeros((640, 640, 3), dtype=np.uint8)
        self._run_detection(warmup)
        logger.info("YOLOX session warmed up.")

    def _run_detection(self, img_orig: np.ndarray):
        img_resized = cv2.resize(img_orig, (self.w_in, self.h_in))
        img_input = img_resized.transpose(2, 0, 1)[np.newaxis].astype(
            self.input_dtype
        )

        with self._inference_lock:
            outputs = self.det_session.run(None, {self.input_name: img_input})

        dets_raw, labels_raw = outputs[0], outputs[1]
        if dets_raw.ndim == 3:
            dets_raw = dets_raw[0]
        if labels_raw.ndim == 2:
            labels_raw = labels_raw[0]
        return dets_raw, labels_raw

    def _filter_detections(self, dets_raw, labels_raw):
        dets, labels = [], []
        for det, cls_id in zip(dets_raw, labels_raw):
            if float(det[4]) < self.args.threshold:
                continue
            dets.append([
                float(det[0]), float(det[1]),
                float(det[2]), float(det[3]),
                float(det[4]),
            ])
            labels.append(int(cls_id))
        if dets:
            return (
                np.array(dets, dtype=np.float32),
                np.array(labels, dtype=np.int32),
            )
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # ------------------------------------------------------------------
    # Predictor (CPU-only)
    # ------------------------------------------------------------------

    def _init_predictor(self) -> None:
        predictor_path = getattr(self.args, 'predictor_model_path', None)
        if predictor_path is None:
            logger.warning(
                "No predictor_model_path — anomaly scoring disabled."
            )
            self.pred_session = None
            return

        logger.info("Creating predictor inference session (CPU)...")
        so = rt.SessionOptions()
        so.log_severity_level = 3
        self.pred_session = rt.InferenceSession(
            predictor_path,
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )
        pred_input = self.pred_session.get_inputs()[0]
        self.pred_input_name = pred_input.name
        self.pred_window_size = pred_input.shape[1]
        self.pred_num_features = pred_input.shape[2]
        self.window_size = self.pred_window_size

        dummy = np.zeros(pred_input.shape, dtype=np.float32)
        self.pred_session.run(None, {self.pred_input_name: dummy})
        logger.info(
            "Predictor warmed up. window=%d, features=%d",
            self.pred_window_size, self.pred_num_features,
        )

    # ------------------------------------------------------------------
    # Feature extraction: [cx, cy, bw, bh, dcx, dcy, dbw, dbh, conf]
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        track_id: int,
        bbox_scaled: list,
        img_w: int,
        img_h: int,
        cam: AnomalyCameraState,
    ) -> np.ndarray:
        """
        Extract [cx, cy, bw, bh, dcx, dcy, dbw, dbh, conf].

        Spatial values normalized by image dimensions.
        Deltas are zero on the first frame a track appears.
        """
        x1, y1, x2, y2, conf, _cls = bbox_scaled

        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        prev = cam.prev_bbox_features.get(track_id)
        if prev is not None:
            dcx = cx - prev[0]
            dcy = cy - prev[1]
            dbw = bw - prev[2]
            dbh = bh - prev[3]
        else:
            dcx = dcy = dbw = dbh = 0.0

        cam.prev_bbox_features[track_id] = (cx, cy, bw, bh)

        return np.array(
            [cx, cy, bw, bh, dcx, dcy, dbw, dbh, conf],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def _score_tracks(self, cam: AnomalyCameraState) -> None:
        if self.pred_session is None:
            return

        for tid, history in cam.track_histories.items():
            if len(history) < self.window_size:
                continue
            window = np.array(
                list(history)[-self.window_size:], dtype=np.float32
            )
            inp = window[np.newaxis]  # (1, window_size, 9)
            result = self.pred_session.run(
                None, {self.pred_input_name: inp}
            )
            mse = float(result[0][0])
            score = 1.0 - np.exp(-mse / self.anomaly_scale)
            cam.anomaly_scores[tid] = score

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, cam_id: str | int = 0
    ) -> tuple[dict, FrameTimings]:
        """
        Returns:
            frame_tracks: {track_id: [x1, y1, x2, y2, conf, cls_id, anomaly_score]}
            timings:       FrameTimings breakdown
        """
        t_total_start = time.perf_counter()

        if cam_id not in self.cameras:
            raise KeyError(f"Unknown camera '{cam_id}'.")
        cam = self.cameras[cam_id]

        h_orig, w_orig = frame.shape[:2]
        sx, sy = w_orig / self.w_in, h_orig / self.h_in

        # --- Detection ---
        t0 = time.perf_counter()
        dets_raw, labels_raw = self._run_detection(frame)
        t_det = time.perf_counter()

        # --- Filtering ---
        dets, labels = self._filter_detections(dets_raw, labels_raw)
        t_filt = time.perf_counter()

        # --- Tracking ---
        tracked = cam.tracker.update(
            dets if len(dets) > 0 else np.empty((0, 5), dtype=np.float32)
        )
        frame_labels = associate_labels_to_tracks(dets, labels, tracked)
        for tid, (cid, score) in frame_labels.items():
            if cid >= 0:
                cam.persistent_tracks[tid] = (cid, score)
        t_track = time.perf_counter()

        # --- Feature extraction ---
        frame_tracks = {}
        active_track_ids = set()

        for track in tracked:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            active_track_ids.add(track_id)
            cls_id, score = cam.persistent_tracks.get(track_id, (-1, 0.0))

            bbox_scaled = [
                int(x1 * sx), int(y1 * sy),
                int(x2 * sx), int(y2 * sy),
                score, cls_id,
            ]

            if track_id not in cam.track_histories:
                cam.track_histories[track_id] = deque(
                    maxlen=self.window_size
                )
            features = self._extract_features(
                track_id, bbox_scaled, w_orig, h_orig, cam
            )
            cam.track_histories[track_id].append(features)

            frame_tracks[track_id] = bbox_scaled

        # Prune dead tracks
        dead_ids = set(cam.track_histories.keys()) - active_track_ids
        for tid in dead_ids:
            cam.track_histories.pop(tid, None)
            cam.anomaly_scores.pop(tid, None)
            cam.prev_bbox_features.pop(tid, None)

        t_feat = time.perf_counter()

        # --- Anomaly scoring ---
        self._score_tracks(cam)
        t_anom = time.perf_counter()

        # Attach anomaly scores
        for track_id in frame_tracks:
            anomaly = cam.anomaly_scores.get(track_id, None)
            frame_tracks[track_id].append(anomaly)

        timings = FrameTimings(
            detection_ms=(t_det - t0) * 1000,
            filtering_ms=(t_filt - t_det) * 1000,
            tracking_ms=(t_track - t_filt) * 1000,
            feature_extraction_ms=(t_feat - t_track) * 1000,
            anomaly_scoring_ms=(t_anom - t_feat) * 1000,
            total_ms=(time.perf_counter() - t_total_start) * 1000,
        )

        return frame_tracks, timings

    def process_batch(
        self, frames: dict[str | int, np.ndarray]
    ) -> dict[str | int, tuple[dict, FrameTimings]]:
        return {
            cam_id: self.process_frame(frame, cam_id)
            for cam_id, frame in frames.items()
        }
    
    def get_config(self) -> dict:
        """Snapshot of currently-tunable parameters."""
        with self._config_lock:
            return {
                "sort_max_age":        self.args.sort_max_age,
                "sort_min_hits":       self.args.sort_min_hits,
                "sort_iou_threshold":  self.args.sort_iou_threshold,
                "detection_threshold": self.args.threshold,
                "anomaly_threshold":   self.anomaly_threshold,
                "anomaly_scale":       self.anomaly_scale,
            }

    def update_config(
        self,
        *,
        sort_max_age:        int   | None = None,
        sort_min_hits:       int   | None = None,
        sort_iou_threshold:  float | None = None,
        detection_threshold: float | None = None,
        anomaly_threshold:   float | None = None,
        anomaly_scale:       float | None = None,
        rebuild_trackers:    bool        = False,
    ) -> dict:
        """
        Update runtime parameters in place. Inference sessions are NOT rebuilt.
        Only the keyword args you pass are changed. Returns the new full config.
        """
        with self._config_lock:
            if sort_max_age is not None:
                if sort_max_age < 1:
                    raise ValueError("sort_max_age must be >= 1")
                self.args.sort_max_age = int(sort_max_age)
            if sort_min_hits is not None:
                if sort_min_hits < 1:
                    raise ValueError("sort_min_hits must be >= 1")
                self.args.sort_min_hits = int(sort_min_hits)
            if sort_iou_threshold is not None:
                if not 0.0 < sort_iou_threshold <= 1.0:
                    raise ValueError("sort_iou_threshold must be in (0, 1]")
                self.args.sort_iou_threshold = float(sort_iou_threshold)

            if detection_threshold is not None:
                if not 0.0 <= detection_threshold <= 1.0:
                    raise ValueError("detection_threshold must be in [0, 1]")
                self.args.threshold = float(detection_threshold)

            if anomaly_threshold is not None:
                self.anomaly_threshold = float(anomaly_threshold)
            if anomaly_scale is not None:
                if anomaly_scale <= 0.0:
                    raise ValueError("anomaly_scale must be > 0")
                self.anomaly_scale = float(anomaly_scale)

            sort_changed = any(
                v is not None
                for v in (sort_max_age, sort_min_hits, sort_iou_threshold)
            )
            if sort_changed or rebuild_trackers:
                self._apply_sort_params(rebuild=rebuild_trackers)

            cfg = self.get_config.__wrapped__(self) if False else {
                "sort_max_age":        self.args.sort_max_age,
                "sort_min_hits":       self.args.sort_min_hits,
                "sort_iou_threshold":  self.args.sort_iou_threshold,
                "detection_threshold": self.args.threshold,
                "anomaly_threshold":   self.anomaly_threshold,
                "anomaly_scale":       self.anomaly_scale,
            }

        logger.info(
            "Runtime config updated: %s (rebuild_trackers=%s)",
            cfg, rebuild_trackers,
        )
        return cfg

    def _apply_sort_params(self, *, rebuild: bool) -> None:
        """Push current sort_* args onto every camera's tracker."""
        for cam_id, cam in self.cameras.items():
            if not rebuild:
                missing = [
                    n for n in self._SORT_PARAMS if not hasattr(cam.tracker, n)
                ]
                if not missing:
                    for name in self._SORT_PARAMS:
                        setattr(
                            cam.tracker,
                            name,
                            getattr(self.args, f"sort_{name}"),
                        )
                    continue
                logger.warning(
                    "FastSort missing %s on camera '%s'; rebuilding tracker "
                    "(track state cleared).",
                    missing, cam_id,
                )
            # Rebuild path (explicit or fallback)
            cam.tracker = FastSort(
                max_age=self.args.sort_max_age,
                min_hits=self.args.sort_min_hits,
                iou_threshold=self.args.sort_iou_threshold,
            )
            cam.persistent_tracks.clear()
            cam.track_histories.clear()
            cam.prev_bbox_features.clear()
            cam.anomaly_scores.clear()