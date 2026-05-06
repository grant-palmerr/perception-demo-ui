#!/usr/bin/env python3

#
# Usage:
# export TIDL_TOOLS_PATH=""
# export SOC=j721e
# python3 inference_test.py --trial trial_name --threshold 0.5

import onnxruntime as rt
import numpy as np
import cv2
import os
import glob
import argparse
import time
import shutil

# ---- nuImages class names (10 classes, matching COCO-converted order) ----
CLASS_NAMES = [
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
]

# Colors for each class (BGR)
CLASS_COLORS = [
    (0, 255, 0),     # car - green
    (0, 165, 255),   # truck - orange
    (0, 255, 255),   # trailer - yellow
    (255, 0, 0),     # bus - blue
    (0, 0, 255),     # construction_vehicle - red
    (255, 255, 0),   # bicycle - cyan
    (255, 0, 255),   # motorcycle - magenta
    (203, 192, 255), # pedestrian - pink
    (0, 128, 255),   # traffic_cone - dark orange
    (128, 128, 128), # barrier - gray
]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX nuImages inference on TDA4VM")
    parser.add_argument("--trial", type=str, required=True, help="The name of the trial")
    parser.add_argument("--model_path", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--prototxt_path", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.prototxt",
                        help="Path to prototxt meta arch file")
    parser.add_argument("--artifacts_folder", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/artifacts",
                        help="Path to compiled TIDL artifacts")
    parser.add_argument("--img_dir", type=str,
                        default="./test_images",
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./output_images",
                        help="Directory to save output images")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection score threshold")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save output images (just print stats)")
    return parser.parse_args()


def create_session(model_path, prototxt_path, artifacts_folder):
    tidl_tools_path = os.environ.get("TIDL_TOOLS_PATH", "")

    infer_options = {
        "artifacts_folder": artifacts_folder,
        "debug_level": 0,
        "object_detection:meta_layers_names_list": prototxt_path,
        "object_detection:meta_arch_type": 6,
    }
    
    # Only pass the key if it actually exists (EVM relies on /usr/lib)
    if tidl_tools_path:
        infer_options["tidl_tools_path"] = tidl_tools_path

    so = rt.SessionOptions()
    EP_list = ['TIDLExecutionProvider', 'CPUExecutionProvider']
    sess = rt.InferenceSession(
        model_path,
        providers=EP_list,
        provider_options=[infer_options, {}],
        sess_options=so
    )
    return sess


def run_inference(sess, img_orig):
    input_details = sess.get_inputs()
    input_name = input_details[0].name
    input_shape = input_details[0].shape  # [1, 3, 640, 640]

    img = cv2.resize(img_orig, (input_shape[3], input_shape[2]))
    img_input = img.transpose(2, 0, 1)  # HWC -> CHW
    img_input = np.expand_dims(img_input, axis=0)  # -> [1, 3, 640, 640]
    img_input = img_input.astype(np.uint8)  # <-- changed from float32 to uint8

    t_start = time.perf_counter()
    outputs = sess.run(None, {input_name: img_input})
    t_end = time.perf_counter()
    infer_time_ms = (t_end - t_start) * 1000

    dets_raw = outputs[0][0]    # [N, 5]
    labels_raw = outputs[1][0]  # [N]

    dets, labels = [], []
    for det, cls_id in zip(dets_raw, labels_raw):
        score = det[4]
        if score <= 0:
            continue
        dets.append([det[0], det[1], det[2], det[3], score])
        labels.append(cls_id)

    if len(dets) > 0:
        dets = np.array(dets)
        labels = np.array(labels)
    else:
        dets = np.empty((0, 5))
        labels = np.empty((0,))

    return dets, labels, infer_time_ms


def draw_detections(img, dets, labels, threshold, input_shape):
    """Draw bounding boxes on image. Returns annotated image and detection count."""
    h_orig, w_orig = img.shape[:2]
    sx = w_orig / input_shape[3]
    sy = h_orig / input_shape[2]

    count = 0
    for det, cls_id in zip(dets, labels):
        if cls_id < 0:
            continue
        x1, y1, x2, y2, score = det
        if score < threshold:
            continue

        cls_id = int(cls_id)
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)

        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        count += 1

    return img, count


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.trial)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Collect images
    test_images = sorted(
        glob.glob(os.path.join(args.img_dir, "*.jpg")) +
        glob.glob(os.path.join(args.img_dir, "*.png")) +
        glob.glob(os.path.join(args.img_dir, "*.jpeg"))
    )
    if not test_images:
        print(f"No images found in {args.img_dir}")
        return

    print(f"Found {len(test_images)} images")

    # Create session
    print("Creating inference session...")
    sess = create_session(args.model_path, args.prototxt_path, args.artifacts_folder)
    input_shape = sess.get_inputs()[0].shape

    # Output directory
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    # Run inference on all images
    total_time = 0
    total_dets = 0

    # Warmup run (first inference is always slower due to setup)
    print("Warmup run...")
    warmup_img = cv2.imread(test_images[0])
    _ = run_inference(sess, warmup_img)

    print(f"\nRunning inference on {len(test_images)} images...\n")

    for img_path in test_images:
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            print(f"  SKIP (can't read): {img_path}")
            continue

        dets, labels, infer_time_ms = run_inference(sess, img_orig)
        total_time += infer_time_ms

        img_out, count = draw_detections(
            img_orig.copy(), dets, labels, args.threshold, input_shape
        )
        total_dets += count

        fname = os.path.basename(img_path)
        print(f"  {fname}: {count} detections, {infer_time_ms:.1f} ms")

        if not args.no_save:
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, img_out)

    # Summary
    n = len(test_images)
    print(f"\n{'='*50}")
    print(f"  Images processed:   {n}")
    print(f"  Total detections:   {total_dets}")
    print(f"  Avg detections/img: {total_dets/n:.1f}")
    print(f"  Avg inference time: {total_time/n:.1f} ms")
    print(f"  Avg FPS:            {1000*n/total_time:.1f}")
    print(f"{'='*50}")

    if not args.no_save:
        print(f"\nOutput images saved to: {output_dir}/")


if __name__ == "__main__":
    main()