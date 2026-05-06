import asyncio
import json
import logging
import base64
import os

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Literal

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.websockets import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.anomaly_tracking_pipeline import AnomalyTrackingPipeline
from backend.dataloader import NuScenesDataLoader, VideoDataLoader

logger = logging.getLogger(__name__)


_SENTINEL = object()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    # Detection model
    model_path: str        = "./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.onnx"
    prototxt_path: str     = "./models/od-custom-yolox_s_lite_nuimages_onnx/yolox_s_lite_nuimages.prototxt"
    artifacts_folder: str  = "./models/od-custom-yolox_s_lite_nuimages_onnx/artifacts"

    # Predictor model (set to None to disable anomaly scoring)
    predictor_model_path: str | None = "./models/anomaly_detection_predictor_model/predictor.onnx"
    predictor_window_size: int       = 16
    anomaly_threshold: float         = 0.5

    # Data source
    data_source: Literal["nuscenes", "video"] = "nuscenes"
    root_dir: str          = "/mnt/nvme/nuscenes-v1.0-mini-subset"
    cameras: list[str]     = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]

    # Tracking
    threshold: float       = 0.2
    sort_max_age: int      = 5
    sort_min_hits: int     = 2
    sort_iou_threshold: float = 0.1

    # Loop / streaming
    loop_sequences: bool   = True
    subscriber_queue_depth: int = 4
    sync_tolerance_us: int = 50_000


# ---------------------------------------------------------------------------
# Broadcast hub
# ---------------------------------------------------------------------------

class BroadcastHub:
    def __init__(self, queue_depth: int = 4):
        self._queue_depth = queue_depth
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_depth)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    def publish(self, frame: dict) -> None:
        logger.debug(
            "publish cam=%s status=%s subs=%d",
            frame.get("cam_id"), frame.get("status"), len(self._subscribers),
        )
        for q in list(self._subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

class DataSourceUpdate(BaseModel):
    data_source: Literal["nuscenes", "video"]
    root_dir: str
    cameras: list[str] | None = None  # None = auto-discover (video) or use existing (nuscenes)


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    config:   PipelineConfig  = field(default_factory=PipelineConfig)
    pipeline: AnomalyTrackingPipeline | None = None
    loader:   NuScenesDataLoader | None = None
    hub:      BroadcastHub    = field(default_factory=BroadcastHub)
    _inference_task: asyncio.Task | None = None

    # Fields that cannot be changed without recreating the TIDL inference
    # session. Mutating these at runtime is unsafe — the model bindings hang.
    IMMUTABLE_FIELDS = frozenset({
        "model_path", "prototxt_path", "artifacts_folder",
        "predictor_model_path", "predictor_window_size",
    })

    def init_pipeline(self) -> None:
        """First-time boot only. Builds the TIDL session — never call this twice."""
        class _Args:
            pass
        args = _Args()
        for k, v in self.config.dict().items():
            setattr(args, k, v)

        self.pipeline = AnomalyTrackingPipeline(args, camera_ids=self.config.cameras)
        self._build_loader()
        self.hub = BroadcastHub(queue_depth=self.config.subscriber_queue_depth)
        logger.info(
            "Pipeline + %s loader initialized (root=%s, cameras=%s)",
            self.config.data_source, self.config.root_dir, self.config.cameras,
        )

    def _build_loader(self) -> None:
        LoaderCls = (
            VideoDataLoader if self.config.data_source == "video"
            else NuScenesDataLoader
        )
        self.loader = LoaderCls.from_local(
            root_dir=self.config.root_dir,
            cameras=self.config.cameras,
        )

    async def apply_runtime_config(self, new_config: PipelineConfig) -> None:
        current  = self.config.dict()
        proposed = new_config.dict()

        bad = [k for k in self.IMMUTABLE_FIELDS if current[k] != proposed[k]]
        if bad:
            raise HTTPException(
                409,
                f"Fields {bad} cannot change at runtime (would require "
                f"recreating the TIDL inference session). Restart the server.",
            )

        loader_changed = any(
            current[k] != proposed[k]
            for k in ("data_source", "root_dir", "cameras")
        )
        hub_changed = (
            current["subscriber_queue_depth"] != proposed["subscriber_queue_depth"]
        )

        # Stop FIRST. Mutating pipeline.cameras while a frame is in flight
        # crashes the executor thread on the next process_frame call.
        if loader_changed:
            await self.stop_inference_loop()

        if self.pipeline is not None:
            self.pipeline.update_config(
                sort_max_age        = new_config.sort_max_age,
                sort_min_hits       = new_config.sort_min_hits,
                sort_iou_threshold  = new_config.sort_iou_threshold,
                detection_threshold = new_config.threshold,
                anomaly_threshold   = new_config.anomaly_threshold,
            )

        self.config = new_config

        if loader_changed:
            self._build_loader()
            self._sync_pipeline_cameras()

        if hub_changed:
            self.hub = BroadcastHub(queue_depth=new_config.subscriber_queue_depth)

        if loader_changed:
            await self.restart_inference_loop()

    _in_flight_future: asyncio.Future | None = None

    async def stop_inference_loop(self) -> None:
        if self._inference_task and not self._inference_task.done():
            self._inference_task.cancel()
        if self._inference_task is not None:
            try:
                await self._inference_task
            except BaseException:
                pass
        if self._in_flight_future is not None and not self._in_flight_future.done():
            try:
                await self._in_flight_future
            except BaseException:
                pass
        self._in_flight_future = None
        self._inference_task   = None

    def _sync_pipeline_cameras(self) -> None:
        """Add/remove cameras on the live pipeline to match config.cameras."""
        if self.pipeline is None:
            return
        existing = set(self.pipeline.cameras.keys())
        target   = set(self.config.cameras)
        for cam in existing - target:
            self.pipeline.remove_camera(cam)
        for cam in target - existing:
            self.pipeline.add_camera(cam)

    async def restart_inference_loop(self) -> None:
        await self.stop_inference_loop()
        self._inference_task = asyncio.create_task(
            _inference_loop(self), name="inference-loop"
        )
        logger.info("Inference loop (re)started.")
    


_state = AppState()


# ---------------------------------------------------------------------------
# Executor work unit
# ---------------------------------------------------------------------------

def _process_and_encode_multi(
    pipeline:   AnomalyTrackingPipeline,
    cam_frames: dict[str, tuple],
) -> dict[str, tuple]:
    results = {}
    for cam_id, (frame, raw_jpg) in cam_frames.items():
        tracks, timings = pipeline.process_frame(frame, cam_id)
        encoded = base64.b64encode(raw_jpg).decode()
        results[cam_id] = (tracks, timings, encoded)
    return results


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

async def _inference_loop(state: AppState) -> None:
    loop = asyncio.get_event_loop()

    while True:
        multi_seq_iter = iter(state.loader.multi_camera_sequence(
            sync_tolerance_us=state.config.sync_tolerance_us,
        ))
        state.pipeline.reset_all()
        state.hub.publish({"status": "sequence_start", "seq_idx": 0})

        prev_future       = None
        prev_meta_per_cam = None

        while True:
            # Pull the next timestep WITHOUT blocking the event loop.
            item = await loop.run_in_executor(None, next, multi_seq_iter, _SENTINEL)
            if item is _SENTINEL:
                break
            cam_data, idx = item

            active = state.pipeline.cameras
            cam_frames = {
                cam_id: (frame, raw_jpg)
                for cam_id, (frame, raw_jpg, _meta, _lidx) in cam_data.items()
                if cam_id in active
            }
            cam_metas = {
                cam_id: meta
                for cam_id, (_frame, _jpg, meta, _lidx) in cam_data.items()
                if cam_id in active
            }
            if not cam_frames:
                continue

            curr_future = loop.run_in_executor(
                None, _process_and_encode_multi, state.pipeline, cam_frames,
            )
            state._in_flight_future = curr_future
            if prev_future is not None:
                results = await prev_future
                _publish_timestep(state.hub, prev_meta_per_cam, results)
            prev_future       = curr_future
            prev_meta_per_cam = cam_metas

        if prev_future is not None:
            results = await prev_future
            _publish_timestep(state.hub, prev_meta_per_cam, results)

        state.hub.publish({"status": "sequence_end", "seq_idx": 0})
        if not state.config.loop_sequences:
            state.hub.publish({"status": "done"})
            return
        logger.info("Inference loop wrapping around.")


def _publish_timestep(
    hub:          BroadcastHub,
    meta_per_cam: dict[str, dict],
    results:      dict[str, tuple],
) -> None:
    for cam_id, (tracks, timings, encoded_jpg) in results.items():
        serializable_tracks = {}
        for tid, vals in tracks.items():
            serializable_tracks[str(tid)] = {
                "bbox": vals[:4],
                "score": vals[4],
                "class_id": int(vals[5]),
                "anomaly_score": vals[6],
            }
        
        hub.publish({
            "cam_id":   cam_id,
            "metadata": meta_per_cam.get(cam_id, {}),
            "tracks":   serializable_tracks,
            "timings":  timings.as_dict(),
            "image":    encoded_jpg,
        })


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    _state.init_pipeline()
    await _state.restart_inference_loop()
    yield
    await _state.stop_inference_loop()
    logger.info("Shutdown.")


app = FastAPI(title="YOLOX-SORT Anomaly Tracking API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    ok = _state._inference_task is not None and not _state._inference_task.done()
    predictor_active = (
        _state.pipeline is not None
        and _state.pipeline.pred_session is not None
    )
    return {
        "status":            "ok",
        "inference_running":  ok,
        "predictor_active":   predictor_active,
        "subscribers":        _state.hub.subscriber_count,
        "cameras":            _state.config.cameras,
    }

@app.get("/config", response_model=PipelineConfig)
def get_config():
    return _state.config

# @app.post("/config", response_model=PipelineConfig)
# async def update_config(new_config: PipelineConfig):
#     """
#     Update runtime configuration on the live pipeline.

#     Model-path fields (model_path, prototxt_path, artifacts_folder,
#     predictor_model_path, predictor_window_size) cannot be changed —
#     they require a server restart because recreating the TIDL inference
#     session at runtime hangs the board.
#     """
#     prev_loader_fields = {
#         k: getattr(_state.config, k)
#         for k in ("data_source", "root_dir", "cameras")
#     }

#     _state.apply_runtime_config(new_config)

#     # Only restart the inference loop if the loader was rebuilt.
#     loader_rebuilt = any(
#         prev_loader_fields[k] != getattr(new_config, k)
#         for k in prev_loader_fields
#     )
#     if loader_rebuilt:
#         await _state.restart_inference_loop()

#     return _state.config


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/tracking")
async def tracking_endpoint(websocket: WebSocket):
    """
    Frame envelope (with anomaly scoring and timings):
        {
            "cam_id": "CAM_FRONT",
            "metadata": {...},
            "tracks": {
                "42": {
                    "bbox": [x1, y1, x2, y2],
                    "score": 0.85,
                    "class_id": 0,
                    "anomaly_score": 0.12
                }
            },
            "timings": {
                "detection_ms": 12.3,
                "filtering_ms": 0.1,
                "tracking_ms": 0.5,
                "feature_extraction_ms": 0.1,
                "anomaly_scoring_ms": 0.3,
                "total_ms": 13.3
            },
            "image": "base64..."
        }
    """
    if _state.pipeline is None or _state.loader is None:
        await websocket.close(code=1011, reason="Pipeline not initialized")
        return

    cameras_param = websocket.query_params.get("cameras")
    camera_filter: set[str] | None = (
        set(cameras_param.split(",")) if cameras_param else None
    )

    await websocket.accept()
    queue = await _state.hub.subscribe()
    logger.info(
        "Subscriber connected (filter=%s). Total: %d",
        camera_filter, _state.hub.subscriber_count,
    )

    stop_flag = asyncio.Event()

    async def _listen_for_stop() -> None:
        try:
            while True:
                msg = await websocket.receive_text()
                if json.loads(msg).get("command") == "stop":
                    stop_flag.set()
                    return
        except (WebSocketDisconnect, Exception):
            stop_flag.set()

    listener_task = asyncio.create_task(_listen_for_stop())

    try:
        while not stop_flag.is_set():
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if camera_filter and "cam_id" in frame and frame["cam_id"] not in camera_filter:
                continue

            await websocket.send_text(json.dumps(frame))
            logger.debug("sent frame cam=%s", frame.get("cam_id"))

            if frame.get("status") == "done":
                break

    except WebSocketDisconnect:
        logger.info("Subscriber disconnected.")
    except Exception as exc:
        logger.exception("Error in subscriber: %s", exc)
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass
    finally:
        await _state.hub.unsubscribe(queue)
        listener_task.cancel()
        logger.info("Subscriber removed. Total: %d", _state.hub.subscriber_count)
        
    
# @app.post("/source", response_model=PipelineConfig)
# async def update_source(update: DataSourceUpdate):
#     """
#     Switch the data source without recreating the inference pipeline.
#     Only the loader is rebuilt; the TIDL session is preserved.

#     For video sources, `cameras` is the list of subdirectory names under
#     `root_dir`. If omitted, every subdirectory is auto-discovered.

#     For nuscenes sources, `cameras` is the list of CAM_* names. If omitted,
#     the current config's cameras are reused.
#     """
#     if not os.path.isdir(update.root_dir):
#         raise HTTPException(404, f"Directory not found: {update.root_dir}")

#     if update.data_source == "video":
#         if update.cameras is None:
#             cameras = sorted(
#                 d for d in os.listdir(update.root_dir)
#                 if os.path.isdir(os.path.join(update.root_dir, d))
#             )
#             if not cameras:
#                 raise HTTPException(
#                     400, f"No video subdirectories found in {update.root_dir}"
#                 )
#         else:
#             cameras = update.cameras
#             missing = [
#                 c for c in cameras
#                 if not os.path.isdir(os.path.join(update.root_dir, c))
#             ]
#             if missing:
#                 raise HTTPException(404, f"Missing video subdirectories: {missing}")
#     else:  # nuscenes
#         cameras = update.cameras if update.cameras is not None else _state.config.cameras
#         sweeps = os.path.join(update.root_dir, "sweeps")
#         if not os.path.isdir(sweeps):
#             raise HTTPException(
#                 400, f"Expected nuScenes 'sweeps/' subdirectory in {update.root_dir}"
#             )

#     new_config = _state.config.copy(update={
#         "data_source": update.data_source,
#         "root_dir":    update.root_dir,
#         "cameras":     cameras,
#     })

#     try:
#         _state.apply_runtime_config(new_config)
#     except Exception as exc:
#         logger.exception("Failed to switch source")
#         raise HTTPException(500, f"Failed to switch source: {exc}")

#     await _state.restart_inference_loop()
#     _state.hub.publish({"status": "source_changed", "data_source": update.data_source})
#     return _state.config

@app.post("/config", response_model=PipelineConfig)
async def update_config(new_config: PipelineConfig):
    await _state.apply_runtime_config(new_config)
    return _state.config


@app.post("/source", response_model=PipelineConfig)
async def update_source(update: DataSourceUpdate):
    
    if not os.path.isdir(update.root_dir):
        raise HTTPException(404, f"Directory not found: {update.root_dir}")
    
    if update.data_source == "video":
        if update.cameras is None:
            cameras = sorted(
                d for d in os.listdir(update.root_dir)
                if os.path.isdir(os.path.join(update.root_dir, d))
            )
            if not cameras:
                raise HTTPException(
                    400, f"No video subdirectories found in {update.root_dir}"
                )
        else:
            cameras = update.cameras
            missing = [
                c for c in cameras
                if not os.path.isdir(os.path.join(update.root_dir, c))
            ]
            if missing:
                raise HTTPException(404, f"Missing video subdirectories: {missing}")
    else:  # nuscenes
        cameras = update.cameras if update.cameras is not None else _state.config.cameras
        sweeps = os.path.join(update.root_dir, "sweeps")
        if not os.path.isdir(sweeps):
            raise HTTPException(
                400, f"Expected nuScenes 'sweeps/' subdirectory in {update.root_dir}"
            )

    # Build a fresh config object instead of mutating in place.
    update_fields = {
        "data_source": update.data_source,
        "root_dir": update.root_dir,
        "cameras": cameras,
    }
    if hasattr(_state.config, "model_copy"):       # Pydantic v2
        _state.config = _state.config.model_copy(update=update_fields)
    else:                                          # Pydantic v1
        _state.config = _state.config.copy(update=update_fields)

    logger.info(
        "Source switched: data_source=%s, root_dir=%s, cameras=%s",
        _state.config.data_source,
        _state.config.root_dir,
        _state.config.cameras,
    )

    try:
        _state.init_pipeline()
    except Exception as exc:
        logger.exception("Failed to initialize new source")
        raise HTTPException(500, f"Failed to initialize source: {exc}")

    await _state.restart_inference_loop()
    _state.hub.publish({"status": "source_changed", "data_source": update.data_source})
    return _state.config