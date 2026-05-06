#!/usr/bin/env python3
#
# Usage:
#   python3 tracking_inference.py --trial trial_name [options]
#
# Setup (on board):
#   git clone https://github.com/ifzhang/ByteTrack.git
#   cp ByteTrack/yolox/tracker/byte_tracker.py .
#   cp ByteTrack/yolox/tracker/kalman_filter.py .
#   cp ByteTrack/yolox/tracker/basetrack.py .
#   cp ByteTrack/yolox/tracker/matching.py .
#   pip install scipy lapx

import onnxruntime as rt
import numpy as np
import cv2
import os
import glob
import argparse
import time
import shutil
import types

from trackers import BYTETracker

# ---- nuImages class names (10 classes) ----
CLASS_NAMES = [
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
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
        description="YOLOX + ByteTrack tracking on nuScenes sweeps (TDA4VM)"
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
                        help="Path to nuScenes CAM_FRONT sweeps directory")
    parser.add_argument("--output_dir", type=str,
                        default="./output_tracking",
                        help="Directory to save annotated frames and video")
    parser.add_argument("--num_frames", type=int, default=120,
                        help="Number of frames to process (0 = all)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="High confidence detection threshold")
    parser.add_argument("--video_fps", type=float, default=12.0,
                        help="FPS for output video (nuScenes front cam ~12Hz)")
    # ByteTrack parameters
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="Frames to keep a lost track alive")
    parser.add_argument("--match_thresh", type=float, default=0.8,
                        help="Matching threshold for track association")
    parser.add_argument("--min_box_area", type=float, default=10.0,
                        help="Minimum bounding box area to track")
    parser.add_argument("--no_save_frames", action="store_true",
                        help="Skip saving individual annotated frames")
    return parser.parse_args()


def make_bytetrack_args(args):
    """Build the SimpleNamespace ByteTracker expects."""
    bt = types.SimpleNamespace()
    bt.track_thresh = args.threshold
    bt.track_buffer = args.track_buffer
    bt.match_thresh = args.match_thresh
    bt.min_box_area = args.min_box_area
    bt.mot20 = False
    return bt


def create_session(model_path, prototxt_path, artifacts_folder):
    infer_options = {
        "artifacts_folder": artifacts_folder,
        "debug_level": 0,
        "object_detection:meta_layers_names_list": prototxt_path,
        "object_detection:meta_arch_type": 6,
    }
    tidl_tools_path = os.environ.get("TIDL_TOOLS_PATH", "")
    if tidl_tools_path:
        infer_options["tidl_tools_path"] = tidl_tools_path

    so = rt.SessionOptions()
    sess = rt.InferenceSession(
        model_path,
        providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
        provider_options=[infer_options, {}],
        sess_options=so,
    )
    return sess


def run_inference(sess, img_orig):
    input_details = sess.get_inputs()
    input_name = input_details[0].name
    input_shape = input_details[0].shape  # [1, 3, H, W]
    h_in, w_in = input_shape[2], input_shape[3]

    img_resized = cv2.resize(img_orig, (w_in, h_in))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, axis=0).astype(np.uint8)

    t = time.perf_counter()
    outputs = sess.run(None, {input_name: img_input})
    infer_ms = (time.perf_counter() - t) * 1000

    dets_raw   = outputs[0]
    labels_raw = outputs[1]
    if dets_raw.ndim == 3:
        dets_raw = dets_raw[0]
    if labels_raw.ndim == 2:
        labels_raw = labels_raw[0]

    return dets_raw, labels_raw, infer_ms, input_shape


def filter_detections(dets_raw, labels_raw, threshold):
    dets, labels = [], []
    for det, cls_id in zip(dets_raw, labels_raw):
        score = float(det[4])
        if score < threshold:
            continue
        dets.append([float(det[0]), float(det[1]),
                     float(det[2]), float(det[3]), score])
        labels.append(int(cls_id))
    if dets:
        return np.array(dets, dtype=np.float32), np.array(labels, dtype=np.int32)
    return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int32)


def associate_labels_to_tracks(dets, labels, online_targets):
    """Match track boxes back to detections by IoU to assign class labels."""
    label_map = {}
    if len(dets) == 0 or not online_targets:
        return label_map
    for target in online_targets:
        tx1, ty1, tx2, ty2 = target.tlbr
        best_iou, best_cls = 0.0, -1
        for det, cls_id in zip(dets, labels):
            dx1, dy1, dx2, dy2 = det[0], det[1], det[2], det[3]
            ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
            ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter == 0:
                continue
            iou = inter / ((tx2-tx1)*(ty2-ty1) + (dx2-dx1)*(dy2-dy1) - inter + 1e-6)
            if iou > best_iou:
                best_iou, best_cls = iou, int(cls_id)
        label_map[target.track_id] = best_cls
    return label_map


def draw_tracks(img, online_targets, persistent_labels, input_shape):
    h_orig, w_orig = img.shape[:2]
    h_in, w_in = input_shape[2], input_shape[3]
    sx, sy = w_orig / w_in, h_orig / h_in

    for target in online_targets:
        x1, y1, x2, y2 = target.tlbr
        x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
        track_id = target.track_id
        cls_id = persistent_labels.get(track_id, -1)
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track_id} {CLASS_NAMES[cls_id]}" if 0 <= cls_id < len(CLASS_NAMES) else f"ID:{track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img, len(online_targets)


def main():
    args = parse_args()

    trial_dir  = os.path.join(args.output_dir, args.trial)
    frames_dir = os.path.join(trial_dir, "frames")
    video_path = os.path.join(trial_dir, f"{args.trial}_tracking.mp4")
    if os.path.exists(trial_dir):
        shutil.rmtree(trial_dir)
    os.makedirs(frames_dir, exist_ok=True)

    all_images = sorted(
        glob.glob(os.path.join(args.sweeps_dir, "*.jpg")) +
        glob.glob(os.path.join(args.sweeps_dir, "*.png"))
    )
    if not all_images:
        print(f"No images found in {args.sweeps_dir}")
        return
    if args.num_frames > 0:
        all_images = all_images[:args.num_frames]
    print(f"Found {len(all_images)} frames to process")

    print("Creating inference session...")
    sess = create_session(args.model_path, args.prototxt_path, args.artifacts_folder)
    print(f"Model input shape: {sess.get_inputs()[0].shape}")
    print(f"Model input type:  {sess.get_inputs()[0].type}")

    print("Warmup run...")
    _, _, _, input_shape = run_inference(sess, cv2.imread(all_images[0]))
    print("Warmup done.\n")

    tracker = BYTETracker(make_bytetrack_args(args), frame_rate=args.video_fps)
    persistent_labels = {}

    total_infer_ms = total_track_ms = total_draw_ms = total_io_ms = 0.0
    total_tracks = 0
    frame_times = []
    video_writer = None

    print(f"{'Frame':>6}  {'File':<50}  {'Dets':>4}  {'Tracks':>6}  "
          f"{'Infer ms':>8}  {'Track ms':>8}")
    print("-" * 100)

    for frame_idx, img_path in enumerate(all_images):
        t_frame = time.perf_counter()

        # Load
        t_io = time.perf_counter()
        img_orig = cv2.imread(img_path)
        io_ms = (time.perf_counter() - t_io) * 1000
        if img_orig is None:
            print(f"  SKIP: {img_path}")
            continue

        # Inference
        dets_raw, labels_raw, infer_ms, input_shape = run_inference(sess, img_orig)
        dets, labels = filter_detections(dets_raw, labels_raw, args.threshold)
        h_in, w_in = input_shape[2], input_shape[3]

        # ByteTrack
        t_track = time.perf_counter()
        online_targets = tracker.update(
            dets if len(dets) > 0 else np.empty((0, 5), dtype=np.float32),
            [h_in, w_in], [h_in, w_in]
        )
        track_ms = (time.perf_counter() - t_track) * 1000

        # Label association
        frame_labels = associate_labels_to_tracks(dets, labels, online_targets)
        for tid, cid in frame_labels.items():
            if cid >= 0:
                persistent_labels[tid] = cid

        # Draw
        t_draw = time.perf_counter()
        img_out = img_orig.copy()
        img_out, n_tracks = draw_tracks(img_out, online_targets, persistent_labels, input_shape)
        cv2.putText(img_out,
                    f"Frame {frame_idx+1}/{len(all_images)}  Infer: {infer_ms:.1f}ms  Tracks: {n_tracks}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        draw_ms = (time.perf_counter() - t_draw) * 1000

        # Save
        t_io2 = time.perf_counter()
        if not args.no_save_frames:
            cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}_{os.path.basename(img_path)}"), img_out)
        if video_writer is None:
            h, w = img_out.shape[:2]
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                           args.video_fps, (w, h))
        video_writer.write(img_out)
        io_ms += (time.perf_counter() - t_io2) * 1000

        total_infer_ms += infer_ms
        total_track_ms += track_ms
        total_draw_ms  += draw_ms
        total_io_ms    += io_ms
        total_tracks   += n_tracks
        frame_times.append(time.perf_counter() - t_frame)

        print(f"{frame_idx+1:>6}  {os.path.basename(img_path)[:50]:<50}  {len(dets):>4}  "
              f"{n_tracks:>6}  {infer_ms:>8.1f}  {track_ms:>8.2f}")

    if video_writer:
        video_writer.release()

    n = len(frame_times)
    if n == 0:
        print("No frames processed.")
        return

    avg_infer  = total_infer_ms / n
    avg_track  = total_track_ms / n
    avg_draw   = total_draw_ms  / n
    avg_io     = total_io_ms    / n
    avg_total  = sum(frame_times) / n * 1000
    avg_stream = avg_infer + avg_track + avg_draw

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
    print(f"\nOutput saved to: {trial_dir}/")
    print(f"  Annotated frames: {frames_dir}/")
    print(f"  Video:            {video_path}")


if __name__ == "__main__":
    main()
