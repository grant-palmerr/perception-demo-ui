"""
Data loader for nuScenes camera sweeps.

Supports two source modes:
  - "local":  frames already on disk under a root directory
  - "azure":  frames in Azure Blob Storage, downloaded in chunks

Usage:
    # --- Local, single-camera ---
    loader = NuScenesDataLoader.from_local(
        root_dir="/data/nuscenes",
        cameras=["CAM_FRONT"],
    )

    for sequence in loader.sequences():
        tracker.reset()
        for frame, raw_jpg, metadata, idx in sequence:
            tracks = pipeline.process_frame(frame)

    # --- Local, multi-camera ---
    loader = NuScenesDataLoader.from_local(
        root_dir="/data/nuscenes",
        cameras=["CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT"],
    )

    mc = loader.multi_camera_sequence(reference_camera="CAM_FRONT")
    for timestep, global_idx in mc:
        # timestep: Dict[str, Tuple[np.ndarray, bytes, dict, int]]
        frame_front, raw_jpg, meta, local_idx = timestep["CAM_FRONT"]
        tracks = pipeline.process_frame(frame_front)

    # --- Azure ---
    loader = NuScenesDataLoader.from_azure(
        container_url="https://<account>.blob.core.windows.net/<container>",
        cameras=["CAM_FRONT", "CAM_BACK"],
        chunk_size=200,
        local_cache_dir="/tmp/nuscenes_cache",
    )

    for sequence in loader.sequences():
        tracker.reset()
        for frame, raw_jpg, metadata, idx in sequence:
            tracks = pipeline.process_frame(frame)
"""

import bisect
import glob
import logging
import os
import re
import shutil
import tempfile
import threading
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("NuScenesDataLoader")

# All nuScenes cameras
ALL_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Pattern: <scene_token>__<CAMERA>__<timestamp_us>.jpg
_NUSCENES_FNAME_RE = re.compile(
    r"^(?P<scene>[^_]+(?:-[^_]+)*)__(?P<camera>CAM_[A-Z_]+)__(?P<timestamp>\d+)\.jpg$"
)


def parse_nuscenes_filename(filename: str) -> Optional[Dict[str, str]]:
    """Extract scene token, camera name, and timestamp from a nuScenes sweep filename."""
    m = _NUSCENES_FNAME_RE.match(filename)
    if m is None:
        return None
    return {
        "scene": m.group("scene"),
        "camera": m.group("camera"),
        "timestamp_us": int(m.group("timestamp")),
    }


@dataclass
class FrameMetadata:
    """Metadata yielded alongside each frame."""
    filename: str
    camera: str
    scene: str
    timestamp_us: int
    source_path: str  # full path on disk (or blob name)
    image_width: int
    image_height: int

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "camera": self.camera,
            "scene": self.scene,
            "timestamp_us": self.timestamp_us,
            "source_path": self.source_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


# ──────────────────────────────────────────────
#  FrameSequence: one camera's ordered frames
# ──────────────────────────────────────────────

class FrameSequence:
    """
    An ordered, lazily-loaded sequence of frames for a single camera.

    Iterating yields (frame: np.ndarray, raw_jpg: bytes, metadata: dict, index: int).
    Random access via __getitem__ is also supported.

    A background prefetch thread reads and decodes images ahead of the consumer.
    If the consumer exits early (break / exception), the loader thread is signalled
    to stop via a threading.Event; queue.put() uses a timeout so the thread will
    eventually notice the stop event even if the queue is full.
    """

    def __init__(self, camera: str, file_paths: List[str], prefetch: int = 4):
        """
        Args:
            camera:     Camera name, e.g. "CAM_FRONT".
            file_paths: Absolute paths to frame images on local disk, in any order
                        (will be sorted by timestamp).
            prefetch:   Number of decoded frames to buffer ahead of the consumer.
        """
        self.camera = camera
        self.prefetch = prefetch

        self._entries: List[Tuple[int, str, FrameMetadata]] = []
        image_w, image_h = self._probe_dimensions(file_paths)

        for p in file_paths:
            fname = os.path.basename(p)
            parsed = parse_nuscenes_filename(fname)
            if parsed is None:
                logger.warning("Skipping non-nuScenes file: %s", fname)
                continue
            meta = FrameMetadata(
                filename=fname,
                camera=parsed["camera"],
                scene=parsed["scene"],
                timestamp_us=parsed["timestamp_us"],
                source_path=p,
                image_width=image_w,
                image_height=image_h,
            )
            self._entries.append((parsed["timestamp_us"], p, meta))

        self._entries.sort(key=lambda e: e[0])
        logger.info(
            "FrameSequence [%s]: %d frames, ts range [%s -> %s]",
            camera,
            len(self._entries),
            self._entries[0][0] if self._entries else "N/A",
            self._entries[-1][0] if self._entries else "N/A",
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def timestamps(self) -> List[int]:
        """Sorted list of timestamps (µs) for all frames in this sequence."""
        return [e[0] for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Iteration (prefetch thread)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[np.ndarray, bytes, dict, int]]:
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def _loader() -> None:
            try:
                for idx, (_, path, meta) in enumerate(self._entries):
                    if stop.is_set():
                        return
                    try:
                        raw_jpg = Path(path).read_bytes()
                        frame = cv2.imdecode(
                            np.frombuffer(raw_jpg, np.uint8), cv2.IMREAD_COLOR
                        )
                    except Exception:
                        logger.exception("Failed to read: %s", path)
                        continue
                    if frame is None:
                        logger.error("Failed to decode: %s", path)
                        continue
                    while not stop.is_set():
                        try:
                            q.put(
                                (frame, raw_jpg, meta.as_dict(), idx), timeout=0.1
                            )
                            break
                        except queue.Full:
                            continue
            finally:
                # Best-effort sentinel: only deliver if the consumer is still
                # alive (stop not set). If stop is set, the consumer's finally
                # has already given up on us — putting None into a full queue
                # would deadlock t.join().
                while not stop.is_set():
                    try:
                        q.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        t = threading.Thread(target=_loader, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield item
        finally:
            stop.set()
            t.join()

    # ------------------------------------------------------------------
    # Random access
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, bytes, dict, int]:
        _, path, meta = self._entries[idx]
        frame = cv2.imread(path)
        if frame is None:
            raise IOError(f"Failed to read image: {path}")
        with open(path, "rb") as f:
            raw_jpg = f.read()
        return frame, raw_jpg, meta.as_dict(), idx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_dimensions(file_paths: List[str]) -> Tuple[int, int]:
        """Read a single image to determine (width, height) for the sequence."""
        for p in file_paths:
            img = cv2.imread(p)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
        raise ValueError("Could not read any image to determine dimensions.")


# ──────────────────────────────────────────────────────────────────────
#  MultiCameraSequence: timestep-aligned multi-camera iterator
# ──────────────────────────────────────────────────────────────────────

# One timestep: { cam_id -> (frame, raw_jpg, meta_dict, local_index) }
TimestepDict = Dict[str, Tuple[np.ndarray, bytes, dict, int]]


class MultiCameraSequence:
    """
    Presents N per-camera FrameSequences as a single timestep-aligned iterator.

    nuScenes sweeps are recorded at ~12 Hz but cameras are not perfectly
    synchronised.  We use nearest-neighbour matching with a configurable
    tolerance window: for each reference-camera timestamp (default CAM_FRONT),
    every other camera contributes the frame whose timestamp is closest,
    provided it falls within `sync_tolerance_us` microseconds.  Timesteps
    where any required camera has no match within tolerance are skipped.

    Yields (timestep_dict: TimestepDict, global_index: int).

    If only one camera is configured the overhead is zero — the iterator
    reduces to a pass-through of that camera's FrameSequence.
    """

    def __init__(
        self,
        sequences: Dict[str, FrameSequence],
        reference_camera: Optional[str] = None,
        sync_tolerance_us: int = 50_000,  # 50 ms — well within one 12-Hz frame
    ):
        if not sequences:
            raise ValueError("At least one camera sequence is required.")

        self._sequences = sequences
        self._sync_tolerance = sync_tolerance_us

        # Pick reference camera: prefer CAM_FRONT, otherwise first alphabetically.
        if reference_camera and reference_camera in sequences:
            self._reference = reference_camera
        elif "CAM_FRONT" in sequences:
            self._reference = "CAM_FRONT"
        else:
            self._reference = sorted(sequences)[0]

        logger.info(
            "MultiCameraSequence: %d cameras, reference=%s, tolerance=%d µs",
            len(sequences),
            self._reference,
            sync_tolerance_us,
        )

    def __len__(self) -> int:
        """Upper bound: number of reference-camera frames."""
        return len(self._sequences[self._reference])

    def __iter__(self) -> Iterator[Tuple[TimestepDict, int]]:
        if len(self._sequences) == 1:
            # Fast path: single camera, no sync overhead.
            cam_id = next(iter(self._sequences))
            for frame, raw_jpg, meta, idx in self._sequences[cam_id]:
                yield {cam_id: (frame, raw_jpg, meta, idx)}, idx
            return

        # Build a sorted (timestamp, entry_index) list for every non-reference
        # camera so we can do O(log n) nearest-neighbour lookup per tick.
        ts_index: Dict[str, List[Tuple[int, int]]] = {}
        for cam_id, seq in self._sequences.items():
            if cam_id == self._reference:
                continue
            ts_index[cam_id] = [(ts, i) for i, ts in enumerate(seq.timestamps)]
            # timestamps property already returns a sorted list, so no extra sort needed.

        skipped = 0
        global_idx = 0

        for ref_frame, ref_jpg, ref_meta, ref_local_idx in self._sequences[self._reference]:
            ref_ts = ref_meta["timestamp_us"]
            timestep: TimestepDict = {
                self._reference: (ref_frame, ref_jpg, ref_meta, ref_local_idx)
            }

            skip_timestep = False
            for cam_id, ts_list in ts_index.items():
                nearest_entry_idx, delta = self._nearest(ts_list, ref_ts)
                if delta > self._sync_tolerance:
                    logger.debug(
                        "Skipping timestep %d: %s nearest frame is %d µs away "
                        "(tolerance %d µs)",
                        ref_ts,
                        cam_id,
                        delta,
                        self._sync_tolerance,
                    )
                    skip_timestep = True
                    break
                frame, raw_jpg, meta, _ = self._sequences[cam_id][nearest_entry_idx]
                timestep[cam_id] = (frame, raw_jpg, meta, nearest_entry_idx)

            if skip_timestep:
                skipped += 1
                continue

            yield timestep, global_idx
            global_idx += 1

        if skipped:
            logger.info(
                "MultiCameraSequence: skipped %d timesteps outside sync tolerance.",
                skipped,
            )

    @staticmethod
    def _nearest(ts_list: List[Tuple[int, int]], target: int) -> Tuple[int, int]:
        """
        Binary search for the entry whose timestamp is closest to `target`.
        Returns (entry_index, abs_delta_us).
        """
        pos = bisect.bisect_left(ts_list, (target, 0))
        best_idx, best_delta = ts_list[0][1], abs(ts_list[0][0] - target)
        for i in range(max(0, pos - 1), min(len(ts_list), pos + 2)):
            delta = abs(ts_list[i][0] - target)
            if delta < best_delta:
                best_delta = delta
                best_idx = ts_list[i][1]
        return best_idx, best_delta


# ──────────────────────────────────────────────
#  Azure chunked download helper
# ──────────────────────────────────────────────

class _AzureChunkedDownloader:
    """Downloads blobs in chunks to a local cache directory, then cleans up."""

    def __init__(self, container_url: str, chunk_size: int = 200, cache_dir: Optional[str] = None):
        from azure.storage.blob import ContainerClient
        self.client = ContainerClient.from_container_url(container_url)
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="nuscenes_cache_")
        os.makedirs(self.cache_dir, exist_ok=True)

    def list_blobs_for_camera(self, camera: str) -> List[str]:
        """Return sorted blob names under sweeps/<camera>/."""
        prefix = f"sweeps/{camera}/"
        blob_names = []
        for blob in self.client.list_blobs(name_starts_with=prefix):
            if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                blob_names.append(blob.name)
        blob_names.sort()
        logger.info("Azure: found %d blobs for %s", len(blob_names), camera)
        return blob_names

    def download_chunk(self, blob_names: List[str]) -> List[str]:
        """Download a list of blobs to cache_dir. Returns local file paths."""
        local_paths = []
        for name in blob_names:
            fname = os.path.basename(name)
            local_path = os.path.join(self.cache_dir, fname)
            if not os.path.exists(local_path):
                blob_data = self.client.download_blob(name).readall()
                with open(local_path, "wb") as f:
                    f.write(blob_data)
            local_paths.append(local_path)
        return local_paths

    def clear_cache(self):
        """Remove all cached files."""
        for f in glob.glob(os.path.join(self.cache_dir, "*")):
            os.remove(f)

    def cleanup(self):
        """Remove the entire cache directory."""
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)


# ──────────────────────────────────────────────
#  Chunked Azure sequence (lazy download + lazy read)
# ──────────────────────────────────────────────

class AzureFrameSequence:
    """
    Like FrameSequence but downloads from Azure in chunks.

    Each chunk is downloaded to disk, iterated lazily, then cleaned up
    before the next chunk is fetched.  This keeps both storage and
    memory usage bounded.

    Yields (frame: np.ndarray, raw_jpg: bytes, metadata: dict, global_index: int).

    Note: random access (__getitem__) and .timestamps are not supported.
    Use NuScenesDataLoader.multi_camera_sequence() only with local sources.
    """

    def __init__(
        self,
        camera: str,
        blob_names: List[str],
        downloader: _AzureChunkedDownloader,
    ):
        self.camera = camera
        self.blob_names = blob_names
        self._downloader = downloader

    def __len__(self) -> int:
        return len(self.blob_names)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, bytes, dict, int]]:
        chunk_size = self._downloader.chunk_size
        global_idx = 0

        for chunk_start in range(0, len(self.blob_names), chunk_size):
            chunk_blobs = self.blob_names[chunk_start: chunk_start + chunk_size]
            logger.info(
                "Azure [%s]: downloading chunk %d-%d / %d",
                self.camera,
                chunk_start,
                chunk_start + len(chunk_blobs) - 1,
                len(self.blob_names),
            )
            local_paths = self._downloader.download_chunk(chunk_blobs)
            seq = FrameSequence(self.camera, local_paths)
            try:
                # Unpack all four values that FrameSequence.__iter__ yields.
                for frame, raw_jpg, meta, _local_idx in seq:
                    yield frame, raw_jpg, meta, global_idx
                    global_idx += 1
            finally:
                # Always clean up the chunk, even if iteration is interrupted.
                self._downloader.clear_cache()


# ──────────────────────────────────────────────
#  NuScenesDataLoader: top-level orchestrator
# ──────────────────────────────────────────────

class NuScenesDataLoader:
    """
    Discovers and iterates over per-camera frame sequences.

    Use the class methods `from_local()` or `from_azure()` to construct.

    Single-camera usage:
        Call `sequences()` to iterate — each yielded sequence represents one
        camera and should correspond to one tracker lifetime (reset between).

    Multi-camera usage (local only):
        Call `multi_camera_sequence()` to get a timestep-aligned
        MultiCameraSequence.  Azure sources are not supported for multi-camera
        because AzureFrameSequence does not expose random access or timestamps.
    """

    def __init__(self):
        # Keyed by camera name so multi_camera_sequence() can look up by cam_id.
        self._sequences: Dict[str, object] = {}
        self._downloader: Optional[_AzureChunkedDownloader] = None

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_local(
        cls,
        root_dir: str,
        cameras: Optional[List[str]] = None,
    ) -> "NuScenesDataLoader":
        """
        Load frames already on disk.

        Expected layout:
            root_dir/
              sweeps/
                CAM_FRONT/
                  <scene>__CAM_FRONT__<timestamp>.jpg
                CAM_BACK/
                  ...
        """
        loader = cls()
        cameras = cameras or ALL_CAMERAS
        sweeps_dir = os.path.join(root_dir, "sweeps")

        for cam in cameras:
            cam_dir = os.path.join(sweeps_dir, cam)
            if not os.path.isdir(cam_dir):
                logger.warning("Camera directory not found: %s", cam_dir)
                continue
            paths = glob.glob(os.path.join(cam_dir, "*.jpg"))
            if not paths:
                logger.warning("No frames found in %s", cam_dir)
                continue
            loader._sequences[cam] = FrameSequence(cam, paths)

        logger.info(
            "NuScenesDataLoader (local): %d sequences loaded", len(loader._sequences)
        )
        return loader

    @classmethod
    def from_azure(
        cls,
        container_url: str,
        cameras: Optional[List[str]] = None,
        chunk_size: int = 200,
        local_cache_dir: Optional[str] = None,
    ) -> "NuScenesDataLoader":
        """
        Stream frames from Azure Blob Storage, downloading in chunks.

        Args:
            container_url:   Full URL to the Azure Blob container, e.g.
                             "https://<account>.blob.core.windows.net/<container>"
            cameras:         Camera names to load (default: all six).
            chunk_size:      Number of frames to download per batch.
            local_cache_dir: Directory for temporary downloads (default: tmpdir).
        """
        loader = cls()
        cameras = cameras or ALL_CAMERAS

        loader._downloader = _AzureChunkedDownloader(
            container_url=container_url,
            chunk_size=chunk_size,
            cache_dir=local_cache_dir,
        )

        for cam in cameras:
            blob_names = loader._downloader.list_blobs_for_camera(cam)
            if not blob_names:
                logger.warning("No blobs found for camera: %s", cam)
                continue
            loader._sequences[cam] = AzureFrameSequence(cam, blob_names, loader._downloader)

        logger.info(
            "NuScenesDataLoader (azure): %d sequences discovered", len(loader._sequences)
        )
        return loader

    # ── Iteration ─────────────────────────────────────────────────────

    def sequences(self) -> Iterator:
        """Yield each camera sequence. Reset your tracker between sequences."""
        yield from self._sequences.values()

    def multi_camera_sequence(
        self,
        reference_camera: Optional[str] = None,
        sync_tolerance_us: int = 50_000,
    ) -> MultiCameraSequence:
        """
        Return a MultiCameraSequence that yields one timestep-aligned dict per tick.

        Only supported for local sources (from_local).  Azure sequences do not
        expose the random access and timestamp index that sync alignment requires.

        Args:
            reference_camera:  Camera to use as the timestamp spine.  Defaults
                               to CAM_FRONT if present, otherwise the first
                               camera alphabetically.
            sync_tolerance_us: Maximum timestamp delta (µs) between the reference
                               frame and a matching frame on another camera.
                               Timesteps where any camera exceeds this are skipped.
                               Default: 50 000 µs (50 ms).

        Returns:
            MultiCameraSequence yielding (TimestepDict, global_index) pairs.

        Raises:
            NotImplementedError: If any sequence is an AzureFrameSequence.
        """
        local_seqs: Dict[str, FrameSequence] = {}
        for cam_id, seq in self._sequences.items():
            if not isinstance(seq, FrameSequence):
                raise NotImplementedError(
                    f"multi_camera_sequence() is not supported for Azure sources "
                    f"(camera '{cam_id}' is an AzureFrameSequence). "
                    "Download the data locally first, then use from_local()."
                )
            local_seqs[cam_id] = seq

        return MultiCameraSequence(
            local_seqs,
            reference_camera=reference_camera,
            sync_tolerance_us=sync_tolerance_us,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_sequences(self) -> int:
        return len(self._sequences)

    def total_frames(self) -> int:
        """Total frame count across all sequences."""
        return sum(len(s) for s in self._sequences.values())

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self):
        """Remove any temporary cache directories (Azure mode)."""
        if self._downloader is not None:
            self._downloader.cleanup()

    def __del__(self):
        self.cleanup()
        

# ──────────────────────────────────────────────
#  Video frame sequence (numbered jpgs)
# ──────────────────────────────────────────────

class VideoFrameSequence(FrameSequence):
    """
    Single-video frame sequence for directories of numbered jpgs.

    Frames are sorted by the LAST integer in the filename stem, so:
      0001.jpg, 0002.jpg, ...           ✓
      frame_0001.jpg, frame_0002.jpg    ✓
      1.jpg, 2.jpg, ..., 10.jpg         ✓ (numeric, not lexicographic)

    Synthesises ~12 Hz timestamps (idx * 83_333 µs) so that
    MultiCameraSequence's nearest-neighbour matching aligns videos by
    frame index without any code changes.
    """

    def __init__(self, video_name: str, file_paths: List[str], prefetch: int = 4):
        # Skip parent __init__: it parses nuScenes filenames.
        self.camera = video_name
        self.prefetch = prefetch

        sorted_paths = sorted(file_paths, key=self._frame_number)
        image_w, image_h = self._probe_dimensions(sorted_paths)

        self._entries: List[Tuple[int, str, FrameMetadata]] = []
        for idx, p in enumerate(sorted_paths):
            fname = os.path.basename(p)
            meta = FrameMetadata(
                filename=fname,
                camera=video_name,
                scene=video_name,
                timestamp_us=idx * 83_333,  # synthetic ~12 Hz cadence
                source_path=p,
                image_width=image_w,
                image_height=image_h,
            )
            self._entries.append((idx * 83_333, p, meta))

        logger.info(
            "VideoFrameSequence [%s]: %d frames",
            video_name, len(self._entries),
        )

    @staticmethod
    def _frame_number(path: str) -> int:
        """Extract the last integer from the filename stem."""
        stem = Path(path).stem
        matches = re.findall(r"\d+", stem)
        if not matches:
            raise ValueError(f"No frame number found in filename: {path}")
        return int(matches[-1])


# ──────────────────────────────────────────────
#  VideoDataLoader: top-level orchestrator
# ──────────────────────────────────────────────

class VideoDataLoader:
    """
    Loads frames from a directory of video subdirectories.

    Expected layout:
        root_dir/
          video_a/
            0000.jpg
            0001.jpg
            ...
          video_b/
            ...

    Each subdirectory is treated as one independent "camera"/stream.
    Calling multi_camera_sequence() aligns them by frame index, so the
    FastAPI inference loop runs unchanged — just point `cameras` at the
    subdirectory names you want to process.

    Notes:
      - Videos of different lengths: timesteps past the shortest video
        get dropped automatically by MultiCameraSequence's tolerance check.
      - Audio, metadata, true timestamps: not supported. timestamp_us in
        the per-frame metadata is synthetic (frame_idx * 83_333 µs).
    """

    def __init__(self):
        self._sequences: Dict[str, VideoFrameSequence] = {}

    @classmethod
    def from_local(
        cls,
        root_dir: str,
        cameras: Optional[List[str]] = None,
    ) -> "VideoDataLoader":
        """
        Args:
            root_dir: Directory containing one subdirectory per video.
            cameras:  Subdirectory names to load. If None, auto-discovers all
                      subdirectories under root_dir (sorted alphabetically).
                      Parameter is named `cameras` rather than `videos` to
                      stay drop-in compatible with NuScenesDataLoader.
        """
        loader = cls()

        if cameras is None:
            cameras = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )

        for cam in cameras:
            video_dir = os.path.join(root_dir, cam)
            if not os.path.isdir(video_dir):
                logger.warning("Video directory not found: %s", video_dir)
                continue
            paths: List[str] = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                paths.extend(glob.glob(os.path.join(video_dir, ext)))
            if not paths:
                logger.warning("No frames found in %s", video_dir)
                continue
            loader._sequences[cam] = VideoFrameSequence(cam, paths)

        logger.info(
            "VideoDataLoader: %d sequences loaded", len(loader._sequences)
        )
        return loader

    def sequences(self) -> Iterator:
        """Yield each video sequence. Reset your tracker between sequences."""
        yield from self._sequences.values()

    def multi_camera_sequence(
        self,
        reference_camera: Optional[str] = None,
        sync_tolerance_us: int = 50_000,
    ) -> MultiCameraSequence:
        """
        Return a frame-index-aligned multi-video iterator.

        Reuses MultiCameraSequence: because every video's synthetic timestamp
        is `idx * 83_333` µs, frames at the same index across videos match
        with delta=0, while frames at different indices are at least 83_333 µs
        apart — outside the default 50_000 µs tolerance — so cross-index
        matching is impossible.
        """
        return MultiCameraSequence(
            self._sequences,
            reference_camera=reference_camera,
            sync_tolerance_us=sync_tolerance_us,
        )

    @property
    def num_sequences(self) -> int:
        return len(self._sequences)

    def total_frames(self) -> int:
        return sum(len(s) for s in self._sequences.values())

    def cleanup(self):
        pass  # nothing to clean up for local video sources

    def __del__(self):
        self.cleanup()