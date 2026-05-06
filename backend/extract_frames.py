"""
Extract frames from a folder of videos into the subdirectory layout
used by VideoDataLoader.

Input:                          Output:
    videos_dir/                     output_dir/
      video_a.mp4                     video_a/
      video_b.mp4                       000000.jpg
      ...                               000001.jpg
                                        ...
                                      video_b/
                                        ...

Usage:
    python extract_frames.py /path/to/videos /path/to/output
    python extract_frames.py /path/to/videos /path/to/output --stride 2 --quality 90
    python extract_frames.py /path/to/videos /path/to/output --max-frames 500 --force
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

logger = logging.getLogger("extract_frames")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def extract_video(
    video_path: Path,
    output_dir: Path,
    stride: int = 1,
    quality: int = 95,
    max_frames: int | None = None,
) -> int:
    """Extract frames from one video. Returns number of frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open: %s", video_path)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    src_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx % stride == 0:
                out_path = output_dir / f"{written:06d}.jpg"
                cv2.imwrite(str(out_path), frame, encode_params)
                written += 1
                if max_frames is not None and written >= max_frames:
                    break
            src_idx += 1
    finally:
        cap.release()

    logger.info(
        "%s: read %d/%d source frames, wrote %d (stride=%d)",
        video_path.name, src_idx, total, written, stride,
    )
    return written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("videos_dir", type=Path, help="Directory containing input videos.")
    parser.add_argument("output_dir", type=Path, help="Where to write frame subdirectories.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every Nth source frame (default: 1).")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 0-100 (default: 95).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames per video (default: no cap).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing non-empty subdirectories.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if not 0 <= args.quality <= 100:
        parser.error("--quality must be 0-100")
    if not args.videos_dir.is_dir():
        logger.error("Not a directory: %s", args.videos_dir)
        sys.exit(1)

    videos = sorted(
        p for p in args.videos_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        logger.error("No videos found in %s", args.videos_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Found %d videos. Output: %s", len(videos), args.output_dir)

    total_written = 0
    for video in videos:
        out_subdir = args.output_dir / video.stem
        if out_subdir.exists() and any(out_subdir.iterdir()):
            if not args.force:
                logger.info("Skipping %s (exists; use --force to overwrite)", video.name)
                continue
            for f in out_subdir.iterdir():
                f.unlink()

        total_written += extract_video(
            video, out_subdir,
            stride=args.stride,
            quality=args.quality,
            max_frames=args.max_frames,
        )

    logger.info("Done. Total frames written: %d", total_written)


if __name__ == "__main__":
    main()