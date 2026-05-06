#!/usr/bin/env python3
"""
Run a folder of sequential frames through the anomaly tracking pipeline.

Saves annotated frames and a timing report.

Usage:
    python run_evaluation.py \
        --input-dir ./frames/ \
        --output-dir ./output/ \
        --model-path ./models/od-custom-yolox_nano_lite_nuimages_onnx/yolox_nano_lite_nuimages.onnx \
        --prototxt-path ./models/od-custom-yolox_nano_lite_nuimages_onnx/yolox_nano_lite_nuimages.prototxt \
        --artifacts-folder ./models/od-custom-yolox_nano_lite_nuimages_onnx/artifacts \
        --predictor-model-path ./models/predictor.onnx
"""
import argparse
import csv
import glob
import json
import logging
import os
import sys
import time

import cv2
import numpy as np

from backend.anomaly_tracking_pipeline import (
    AnomalyTrackingPipeline,
    CLASS_NAMES,
    FrameTimings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_evaluation")


# ── Colours ──────────────────────────────────────────────────────────────

# Neutral color for tracks that don't have an anomaly score yet (BGR grey)
NO_SCORE_COLOR = (180, 180, 180)


def anomaly_color(score: float) -> tuple[int, int, int]:
    """
    Map a 0–1 anomaly score to a BGR color gradient.

        0.0  → pure green  (0, 255, 0)
        0.5  → yellow      (0, 255, 255)
        1.0  → pure red    (0, 0, 255)
    """
    score = max(0.0, min(1.0, score))
    if score <= 0.5:
        t = score / 0.5
        r = int(255 * t)
        g = 255
    else:
        t = (score - 0.5) / 0.5
        r = 255
        g = int(255 * (1.0 - t))
    return (0, g, r)  # BGR


def draw_annotations(
    frame: np.ndarray,
    tracks: dict,
    anomaly_threshold: float,
) -> np.ndarray:
    """Draw bounding boxes colored by anomaly score on a green→red gradient."""
    annotated = frame.copy()

    for track_id, vals in tracks.items():
        x1, y1, x2, y2, conf, cls_id, anomaly = vals
        cls_id = int(cls_id) if cls_id >= 0 else 0

        # Pick color from gradient or grey if no score yet
        if anomaly is not None:
            color = anomaly_color(anomaly)
            thickness = 2 + int(anomaly * 2)  # 2 at 0.0, up to 4 at 1.0

            # Semi-transparent fill for tracks above threshold
            if anomaly > anomaly_threshold:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                alpha = 0.08 + 0.17 * anomaly  # grows with severity
                cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
        else:
            color = NO_SCORE_COLOR
            thickness = 1

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label text
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "?"
        label = f"#{track_id} {cls_name} {conf:.2f}"
        if anomaly is not None:
            label += f" A:{anomaly:.3f}"

        # Text background pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
        )

        # Adaptive text color: black on bright backgrounds, white on dark
        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA,
        )

    return annotated


def collect_frames(input_dir: str) -> list[str]:
    """Glob image files sorted by name."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
    paths.sort()
    if not paths:
        logger.error("No image files found in %s", input_dir)
        sys.exit(1)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Run frames through the anomaly tracking pipeline."
    )
    # I/O
    parser.add_argument("--input-dir",
                        default="/mnt/nvme/nuscenes-v1.0-mini-subset/sweeps/CAM_FRONT",
                        help="Folder of sequential frames")
    parser.add_argument("--output-dir", default="/mnt/nvme/output_anomaly_tracking",
                        help="Where to save annotated frames and stats")

    # Detection model
    parser.add_argument("--model-path", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--prototxt-path", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.prototxt",
                        help="Path to prototxt meta arch file")
    parser.add_argument("--artifacts-folder", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/artifacts",
                        help="Path to compiled TIDL artifacts")

    # Predictor model (omit to disable)
    parser.add_argument("--predictor-model-path", default='models/anomaly_detection_predictor_model/predictor.onnx')
    parser.add_argument("--predictor-window-size", type=int, default=16)

    # Thresholds
    parser.add_argument("--detection-threshold", type=float, default=0.2)
    parser.add_argument("--anomaly-threshold", type=float, default=0.5)

    # Tracker
    parser.add_argument("--sort-max-age", type=int, default=5)
    parser.add_argument("--sort-min-hits", type=int, default=2)
    parser.add_argument("--sort-iou-threshold", type=float, default=0.1)

    # Misc
    parser.add_argument("--camera-id", default="cam0",
                        help="Camera ID label for single-camera eval")
    parser.add_argument("--skip-save-frames", action="store_true",
                        help="Skip saving annotated frames (timing only)")

    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    if not args.skip_save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    frame_paths = collect_frames(args.input_dir)[:300]
    logger.info("Found %d frames in %s", len(frame_paths), args.input_dir)

    # Build pipeline args
    class PipelineArgs:
        pass
    pargs = PipelineArgs()
    pargs.model_path          = args.model_path
    pargs.prototxt_path       = args.prototxt_path
    pargs.artifacts_folder    = args.artifacts_folder
    pargs.predictor_model_path = args.predictor_model_path
    pargs.predictor_window_size = args.predictor_window_size
    pargs.anomaly_threshold   = args.anomaly_threshold
    pargs.threshold           = args.detection_threshold
    pargs.sort_max_age        = args.sort_max_age
    pargs.sort_min_hits       = args.sort_min_hits
    pargs.sort_iou_threshold  = args.sort_iou_threshold

    logger.info("Initializing pipeline...")
    pipeline = AnomalyTrackingPipeline(
        pargs, camera_ids=[args.camera_id]
    )
    logger.info("Pipeline ready.")

    # ── Process frames ───────────────────────────────────────────────
    all_timings: list[dict] = []
    all_track_data: list[dict] = []

    for i, fpath in enumerate(frame_paths):
        frame = cv2.imread(fpath)
        if frame is None:
            logger.warning("Could not read %s — skipping.", fpath)
            continue

        tracks, timings = pipeline.process_frame(frame, cam_id=args.camera_id)

        # Annotate and save
        if not args.skip_save_frames:
            annotated = draw_annotations(
                frame, tracks, args.anomaly_threshold
            )
            out_name = f"frame_{i:06d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, out_name), annotated)

        # Collect timing row
        timing_row = {"frame_idx": i, "filename": os.path.basename(fpath)}
        timing_row.update(timings.as_dict())
        timing_row["num_tracks"] = len(tracks)
        timing_row["num_anomalous"] = sum(
            1 for v in tracks.values()
            if v[6] is not None and v[6] > args.anomaly_threshold
        )
        all_timings.append(timing_row)

        # Collect per-frame track data
        frame_record = {
            "frame_idx": i,
            "filename": os.path.basename(fpath),
            "tracks": {
                str(tid): {
                    "bbox": vals[:4],
                    "confidence": vals[4],
                    "class_id": int(vals[5]),
                    "class_name": (
                        CLASS_NAMES[int(vals[5])]
                        if 0 <= int(vals[5]) < len(CLASS_NAMES)
                        else "unknown"
                    ),
                    "anomaly_score": vals[6],
                }
                for tid, vals in tracks.items()
            },
        }
        all_track_data.append(frame_record)

        # Progress
        if (i + 1) % 50 == 0 or i == len(frame_paths) - 1:
            logger.info(
                "Processed %d/%d  (%.1f ms/frame)",
                i + 1, len(frame_paths), timings.total_ms,
            )

    # ── Save timing CSV ──────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "timings.csv")
    fieldnames = list(all_timings[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_timings)
    logger.info("Saved per-frame timings to %s", csv_path)

    # ── Save track data JSON ─────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "tracks.json")
    with open(json_path, "w") as f:
        json.dump(all_track_data, f, indent=2)
    logger.info("Saved track data to %s", json_path)

    # ── Print summary ────────────────────────────────────────────────
    n = len(all_timings)
    if n == 0:
        logger.error("No frames processed.")
        return

    def stat(key):
        vals = [t[key] for t in all_timings]
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals), np.median(vals)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  Evaluation Summary — {n} frames")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append(f"  {'Stage':<25s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Median':>8s}")
    lines.append(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key in [
        "detection_ms", "filtering_ms", "tracking_ms",
        "feature_extraction_ms", "anomaly_scoring_ms", "total_ms",
    ]:
        mean, std, mn, mx, med = stat(key)
        label = key.replace("_ms", "").replace("_", " ").title()
        lines.append(
            f"  {label:<25s} {mean:>7.2f}  {std:>7.2f}  {mn:>7.2f}  {mx:>7.2f}  {med:>7.2f}"
        )

    lines.append("")
    fps_vals = [1000.0 / t["total_ms"] for t in all_timings if t["total_ms"] > 0]
    if fps_vals:
        lines.append(f"  Throughput: {np.mean(fps_vals):.1f} FPS avg, {np.min(fps_vals):.1f} min, {np.max(fps_vals):.1f} max")

    track_counts = [t["num_tracks"] for t in all_timings]
    anomaly_counts = [t["num_anomalous"] for t in all_timings]
    lines.append(f"  Tracks/frame: {np.mean(track_counts):.1f} avg, {max(track_counts)} max")
    lines.append(f"  Anomalous detections: {sum(anomaly_counts)} total across {n} frames")
    lines.append("")

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()