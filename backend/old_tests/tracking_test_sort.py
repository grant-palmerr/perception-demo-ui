#!/usr/bin/env python3
#
# Usage:
#   export TIDL_TOOLS_PATH=""   # not needed on board
#   python3 tracking_test.py --trial trial_name [options]

import onnxruntime as rt
import numpy as np
import cv2
import os
import glob
import argparse
import time
import shutil

from trackers import Sort

# ---- nuImages class names (10 classes) ----
CLASS_NAMES = [
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
]

CLASS_COLORS = [
    (0, 255, 0),       # car
    (0, 165, 255),     # truck
    (0, 255, 255),     # trailer
    (255, 0, 0),       # bus
    (0, 0, 255),       # construction_vehicle
    (255, 255, 0),     # bicycle
    (255, 0, 255),     # motorcycle
    (203, 192, 255),   # pedestrian
    (0, 128, 255),     # traffic_cone
    (128, 128, 128),   # barrier
]

# Distinct colors for track IDs (cycling)
TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 0, 255),
    (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 255),
    (200, 100, 50), (50, 200, 100), (100, 50, 200), (200, 200, 50),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOX + SORT tracking on nuScenes sweeps (TDA4VM)"
    )
    parser.add_argument("--trial", type=str, required=True,
                        help="Name of the trial (used for output folder)")
    parser.add_argument("--model_path", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--prototxt_path", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.prototxt",
                        help="Path to prototxt meta arch file")
    parser.add_argument("--artifacts_folder", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/artifacts",
                        help="Path to compiled TIDL artifacts")
    parser.add_argument("--sweeps_dir", type=str,
                        default="/root/nuscenes-v1.0-mini-subset/sweeps/CAM_FRONT",
                        help="Path to nuScenes sweeps directory for CAM_FRONT")
    parser.add_argument("--output_dir", type=str,
                        default="./output_tracking",
                        help="Directory to save annotated frames and video")
    parser.add_argument("--num_frames", type=int, default=120,
                        help="Number of frames to process (0 = all)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection score threshold")
    parser.add_argument("--video_fps", type=float, default=12.0,
                        help="FPS for output video (nuScenes front cam is ~12Hz)")
    # SORT parameters
    parser.add_argument("--sort_max_age", type=int, default=3,
                        help="SORT: max frames to keep a track alive without detection")
    parser.add_argument("--sort_min_hits", type=int, default=2,
                        help="SORT: min detections before track is confirmed")
    parser.add_argument("--sort_iou_threshold", type=float, default=0.3,
                        help="SORT: IOU threshold for track association")
    parser.add_argument("--no_save_frames", action="store_true",
                        help="Skip saving individual annotated frames")
    return parser.parse_args()


def create_session(model_path, prototxt_path, artifacts_folder):
    tidl_tools_path = os.environ.get("TIDL_TOOLS_PATH", "")

    infer_options = {
        "artifacts_folder": artifacts_folder,
        "debug_level": 0,
        "object_detection:meta_layers_names_list": prototxt_path,
        "object_detection:meta_arch_type": 6,
    }
    if tidl_tools_path:
        infer_options["tidl_tools_path"] = tidl_tools_path

    so = rt.SessionOptions()
    EP_list = ["TIDLExecutionProvider", "CPUExecutionProvider"]
    sess = rt.InferenceSession(
        model_path,
        providers=EP_list,
        provider_options=[infer_options, {}],
        sess_options=so,
    )
    return sess


def run_inference(sess, img_orig):
    """Run TIDL inference. Returns raw detections and inference time."""
    input_details = sess.get_inputs()
    input_name = input_details[0].name
    input_shape = input_details[0].shape  # [1, 3, H, W]
    h_in, w_in = input_shape[2], input_shape[3]

    img_resized = cv2.resize(img_orig, (w_in, h_in))
    img_input = img_resized.transpose(2, 0, 1)          # HWC -> CHW
    img_input = np.expand_dims(img_input, axis=0)       # -> [1, 3, H, W]
    img_input = img_input.astype(np.uint8)

    t_start = time.perf_counter()
    outputs = sess.run(None, {input_name: img_input})
    infer_time_ms = (time.perf_counter() - t_start) * 1000

    # outputs[0]: [N, 5] (x1, y1, x2, y2, score)
    # outputs[1]: [N]    (class_id)
    dets_raw = outputs[0]   # already [N, 5], no batch dim from TIDL meta-arch
    labels_raw = outputs[1] # [N]

    # Handle case where TIDL adds batch dim
    if dets_raw.ndim == 3:
        dets_raw = dets_raw[0]
    if labels_raw.ndim == 2:
        labels_raw = labels_raw[0]

    return dets_raw, labels_raw, infer_time_ms


def filter_detections(dets_raw, labels_raw, threshold):
    """Filter by score threshold and return clean arrays."""
    dets, labels = [], []
    for det, cls_id in zip(dets_raw, labels_raw):
        score = float(det[4])
        if score < threshold:
            continue
        dets.append([float(det[0]), float(det[1]),
                     float(det[2]), float(det[3]), score])
        labels.append(int(cls_id))

    if len(dets) > 0:
        return np.array(dets, dtype=np.float32), np.array(labels, dtype=np.int32)
    return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int32)


def draw_tracks(img, tracked_dets, det_labels, input_shape, threshold):
    """
    Draw tracked bounding boxes with track IDs.

    tracked_dets: [N, 5] array of (x1, y1, x2, y2, track_id) from SORT
    det_labels:   dict mapping track_id -> class_id (best guess from detections)
    """
    h_orig, w_orig = img.shape[:2]
    h_in, w_in = input_shape[2], input_shape[3]
    sx = w_orig / w_in
    sy = h_orig / h_in

    count = 0
    for track in tracked_dets:
        x1, y1, x2, y2, track_id = track
        track_id = int(track_id)

        x1 = int(x1 * sx)
        y1 = int(y1 * sy)
        x2 = int(x2 * sx)
        y2 = int(y2 * sy)

        # Get class for this track if available
        cls_id = det_labels.get(track_id, -1)

        track_color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), track_color, 2)

        # Build label: "ID:N class_name" or just "ID:N"
        if cls_id >= 0 and cls_id < len(CLASS_NAMES):
            label = f"ID:{track_id} {CLASS_NAMES[cls_id]}"
        else:
            label = f"ID:{track_id}"

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), track_color, -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        count += 1

    return img, count


def associate_labels_to_tracks(dets, labels, tracked_dets):
    """
    Build a mapping of track_id -> class_id by matching tracked boxes
    back to the nearest detection box via IoU.
    """
    label_map = {}
    if len(dets) == 0 or len(tracked_dets) == 0:
        return label_map

    for track in tracked_dets:
        tx1, ty1, tx2, ty2, track_id = track
        track_id = int(track_id)
        best_iou = 0.0
        best_cls = -1

        for det, cls_id in zip(dets, labels):
            dx1, dy1, dx2, dy2 = det[0], det[1], det[2], det[3]

            ix1 = max(tx1, dx1)
            iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2)
            iy2 = min(ty2, dy2)

            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue

            area_t = (tx2 - tx1) * (ty2 - ty1)
            area_d = (dx2 - dx1) * (dy2 - dy1)
            iou = inter / (area_t + area_d - inter + 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_cls = int(cls_id)

        label_map[track_id] = best_cls

    return label_map


def main():
    args = parse_args()

    # Prepare output directories
    frames_dir = os.path.join(args.output_dir, args.trial, "frames")
    video_path = os.path.join(args.output_dir, args.trial,
                              f"{args.trial}_tracking.mp4")
    if os.path.exists(os.path.join(args.output_dir, args.trial)):
        shutil.rmtree(os.path.join(args.output_dir, args.trial))
    os.makedirs(frames_dir, exist_ok=True)

    # Collect and sort frames by timestamp (embedded in filename)
    all_images = sorted(glob.glob(os.path.join(args.sweeps_dir, "*.jpg")) +
                        glob.glob(os.path.join(args.sweeps_dir, "*.png")))
    if not all_images:
        print(f"No images found in {args.sweeps_dir}")
        return

    if args.num_frames > 0:
        all_images = all_images[:args.num_frames]

    print(f"Found {len(all_images)} frames in {args.sweeps_dir}")
    print(f"Processing {len(all_images)} frames...")

    # Create TIDL session
    print("Creating inference session...")
    sess = create_session(args.model_path, args.prototxt_path, args.artifacts_folder)
    input_shape = sess.get_inputs()[0].shape
    print(f"Model input shape: {input_shape}")
    print(f"Model input type:  {sess.get_inputs()[0].type}")

    # Warmup
    print("Warmup run...")
    warmup = cv2.imread(all_images[0])
    run_inference(sess, warmup)
    print("Warmup done.\n")

    # Init SORT tracker
    tracker = Sort(
        max_age=args.sort_max_age,
        min_hits=args.sort_min_hits,
        iou_threshold=args.sort_iou_threshold,
    )

    # Persistent label map: track_id -> class_id (survives across frames)
    persistent_labels = {}

    # Stats
    total_infer_ms = 0.0
    total_track_ms = 0.0
    total_draw_ms = 0.0
    total_io_ms = 0.0
    total_tracks = 0
    frame_times = []

    # Video writer (lazy init after first frame)
    video_writer = None

    print(f"{'Frame':>6}  {'File':<55}  {'Dets':>4}  {'Tracks':>6}  {'Infer ms':>8}  {'Track ms':>8}")
    print("-" * 100)

    for frame_idx, img_path in enumerate(all_images):
        t_frame_start = time.perf_counter()

        # --- I/O: load frame from disk ---
        t_io = time.perf_counter()
        img_orig = cv2.imread(img_path)
        io_ms = (time.perf_counter() - t_io) * 1000

        if img_orig is None:
            print(f"  SKIP (unreadable): {img_path}")
            continue

        # --- Detection ---
        dets_raw, labels_raw, infer_ms = run_inference(sess, img_orig)
        dets, labels = filter_detections(dets_raw, labels_raw, args.threshold)

        # --- Tracking ---
        t_track_start = time.perf_counter()
        if len(dets) > 0:
            tracked = tracker.update(dets)
        else:
            tracked = tracker.update(np.empty((0, 5), dtype=np.float32))
        track_ms = (time.perf_counter() - t_track_start) * 1000

        # --- Associate class labels to track IDs ---
        frame_labels = associate_labels_to_tracks(dets, labels, tracked)
        for tid, cid in frame_labels.items():
            if cid >= 0:
                persistent_labels[tid] = cid

        # --- Draw annotations ---
        t_draw = time.perf_counter()
        img_out = img_orig.copy()
        img_out, n_tracks = draw_tracks(
            img_out, tracked, persistent_labels, input_shape, args.threshold
        )
        cv2.putText(img_out,
                    f"Frame {frame_idx+1}/{len(all_images)}  "
                    f"Infer: {infer_ms:.1f}ms  Tracks: {n_tracks}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        draw_ms = (time.perf_counter() - t_draw) * 1000

        # --- I/O: save frame and write video ---
        t_io2 = time.perf_counter()
        if not args.no_save_frames:
            fname = f"{frame_idx:05d}_{os.path.basename(img_path)}"
            cv2.imwrite(os.path.join(frames_dir, fname), img_out)

        if video_writer is None:
            h, w = img_out.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_path, fourcc, args.video_fps, (w, h))
        video_writer.write(img_out)
        io_ms += (time.perf_counter() - t_io2) * 1000

        total_infer_ms += infer_ms
        total_track_ms += track_ms
        total_draw_ms += draw_ms
        total_io_ms += io_ms
        total_tracks += n_tracks
        frame_times.append(time.perf_counter() - t_frame_start)

        fname_short = os.path.basename(img_path)[:55]
        print(f"{frame_idx+1:>6}  {fname_short:<55}  {len(dets):>4}  "
              f"{n_tracks:>6}  {infer_ms:>8.1f}  {track_ms:>8.2f}")

    if video_writer is not None:
        video_writer.release()

    n = len(all_images)
    avg_infer  = total_infer_ms / n
    avg_track  = total_track_ms / n
    avg_draw   = total_draw_ms  / n
    avg_io     = total_io_ms    / n
    avg_total  = sum(frame_times) / n * 1000
    avg_stream = avg_infer + avg_track + avg_draw  # excludes disk I/O

    print(f"\n{'='*60}")
    print(f"  Frames processed:         {n}")
    print(f"")
    print(f"  --- Per-frame breakdown ---")
    print(f"  Avg inference time:       {avg_infer:.1f} ms")
    print(f"  Avg tracking time:        {avg_track:.2f} ms")
    print(f"  Avg draw/annotate time:   {avg_draw:.2f} ms")
    print(f"  Avg disk I/O time:        {avg_io:.1f} ms  (load + save)")
    print(f"  Avg total frame time:     {avg_total:.1f} ms  (includes I/O)")
    print(f"")
    print(f"  --- Streaming estimate ---")
    print(f"  Avg processing time:      {avg_stream:.1f} ms  (no disk I/O)")
    print(f"  Estimated streaming FPS:  {1000/avg_stream:.1f} fps")
    print(f"")
    print(f"  Avg tracks/frame:         {total_tracks/n:.1f}")
    print(f"{'='*60}")
    print(f"\nOutput saved to: {os.path.join(args.output_dir, args.trial)}/")
    print(f"  Annotated frames: {frames_dir}/")
    print(f"  Video:            {video_path}")


if __name__ == "__main__":
    main()