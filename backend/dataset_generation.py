import argparse
from tracking_pipeline import TrackingPipeline
from backend.dataloader import NuScenesDataLoader
from tqdm import tqdm
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOX + SORT tracking on nuScenes sweeps (TDA4VM)"
    )
    parser.add_argument("--model_path", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--prototxt_path", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.prototxt",
                        help="Path to prototxt meta arch file")
    parser.add_argument("--artifacts_folder", type=str,
                        default="./models/od-custom-yolox_s_lite_nuimages_onnx/artifacts",
                        help="Path to compiled TIDL artifacts")
    parser.add_argument("--dataset_folder", type=str,
                        default="./dataset_collection",
                        help="Directory to save annotated frames and video")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection score threshold")
    parser.add_argument("--cameras", nargs='+', default=['CAM_FRONT'],
                        help="List of cameras to use, ex. --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT")
    # SORT parameters
    parser.add_argument("--sort_max_age", type=int, default=3,
                        help="SORT: max frames to keep a track alive without detection")
    parser.add_argument("--sort_min_hits", type=int, default=2,
                        help="SORT: min detections before track is confirmed")
    parser.add_argument("--sort_iou_threshold", type=float, default=0.3,
                        help="SORT: IOU threshold for track association")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.dataset_folder):
        os.makedirs(args.dataset_folder, exist_ok=True)
    
    loader = NuScenesDataLoader.from_azure(
        container_url="https://smuseniordesign.blob.core.windows.net/nuscenes",
        cameras=args.cameras,
        chunk_size=200,
    )

    pipeline = TrackingPipeline(args)
    
    for seq_idx, sequence in enumerate(loader.sequences()):
        pipeline.reset()  # reset tracker between cameras
        frames = []
        for frame, meta, idx in tqdm(sequence, desc=f"Sequence {seq_idx+1}"):
            tracks = pipeline.process_frame(frame)
            frame_data = {
                'metadata': meta,
                'tracks': tracks,
            }
            frames.append(frame_data)

            if idx % 1000 == 0:
                # save results to json file
                with open(os.path.join(args.dataset_folder, f"{meta['camera']}.json"), 'w') as f:
                    json.dump(frames, f)
        
        # save for full sequence to json file
        with open(os.path.join(args.dataset_folder, f"{meta['camera']}.json"), 'w') as f:
            json.dump(frames, f)
            
        

if __name__ == "__main__":
    main()