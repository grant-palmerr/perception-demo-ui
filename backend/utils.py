import os
import onnxruntime as rt

def create_inference_session(model_path, prototxt_path, artifacts_folder):
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

def associate_labels_to_tracks(dets, labels, tracked_dets):
    """
    Build a mapping of track_id -> (class_id, score) by matching tracked boxes
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
        best_score = 0.0
        for det, cls_id in zip(dets, labels):
            dx1, dy1, dx2, dy2, score = det[0], det[1], det[2], det[3], det[4]
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
                best_score = float(score)
        label_map[track_id] = (best_cls, best_score)
    return label_map