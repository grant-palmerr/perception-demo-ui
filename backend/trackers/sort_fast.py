"""
    SORT: A Simple, Online and Realtime Tracker
    Optimized version — batched Kalman filter, vectorized association.

    Original Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    Licensed under GNU GPL v3.
"""
from __future__ import print_function

import numpy as np
import time

# ---------- Verify lap is available (fall back to scipy with warning) ----------
try:
    import lap
    _USE_LAP = True
except ImportError:
    from scipy.optimize import linear_sum_assignment
    _USE_LAP = False
    import warnings
    warnings.warn(
        "lap not installed — falling back to scipy.optimize.linear_sum_assignment. "
        "Install lap (`pip install lap`) for significantly faster assignment.",
        RuntimeWarning
    )


def linear_assignment(cost_matrix):
    if _USE_LAP:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    else:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# ---------- Vectorized IoU (unchanged from original — already batched) ----------

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


# ---------- Coordinate conversions (vectorized for batch use) ----------

def bbox_to_z(bbox):
    """[x1,y1,x2,y2] → [cx, cy, area, aspect_ratio]  — works for (4,) or (N,4)"""
    if bbox.ndim == 1:
        bbox = bbox[None, :]
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    cx = bbox[:, 0] + w / 2.
    cy = bbox[:, 1] + h / 2.
    s = w * h
    r = w / h
    return np.column_stack([cx, cy, s, r])  # (N, 4)


def z_to_bbox(z):
    """[cx, cy, area, aspect_ratio] → [x1, y1, x2, y2]  — works for (4,) or (N,4)"""
    if z.ndim == 1:
        z = z[None, :]
    w = np.sqrt(z[:, 2] * z[:, 3])
    h = z[:, 2] / w
    x1 = z[:, 0] - w / 2.
    y1 = z[:, 1] - h / 2.
    x2 = z[:, 0] + w / 2.
    y2 = z[:, 1] + h / 2.
    return np.column_stack([x1, y1, x2, y2])  # (N, 4)


# ---------- Vectorized association ----------

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked objects (both as bounding boxes).
    Returns (matches, unmatched_detections, unmatched_trackers).
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    if len(matched_indices) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.arange(len(trackers)))

    # Vectorized: filter matches by IoU threshold and find unmatched
    match_ious = iou_matrix[matched_indices[:, 0], matched_indices[:, 1]]
    good = match_ious >= iou_threshold
    matches = matched_indices[good]

    matched_det_set = set(matches[:, 0]) if len(matches) > 0 else set()
    matched_trk_set = set(matches[:, 1]) if len(matches) > 0 else set()

    unmatched_dets = np.array([d for d in range(len(detections)) if d not in matched_det_set], dtype=int)
    unmatched_trks = np.array([t for t in range(len(trackers)) if t not in matched_trk_set], dtype=int)

    return matches, unmatched_dets, unmatched_trks


# ---------- Batched Kalman Filter State ----------

class BatchKalmanState:
    """
    Manages Kalman filter state for ALL tracks simultaneously in stacked arrays.
    Replaces per-track filterpy.KalmanFilter objects.
    """
    def __init__(self):
        # State transition (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # Observation matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # Process noise
        self.Q = np.eye(7, dtype=np.float64)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.

        # Precompute transposes
        self.F_T = self.F.T.copy()
        self.H_T = self.H.T.copy()

        # Identity for covariance update
        self.I7 = np.eye(7, dtype=np.float64)

        # Per-track state arrays (N tracks)
        self.xs = np.empty((0, 7), dtype=np.float64)      # (N, 7)
        self.Ps = np.empty((0, 7, 7), dtype=np.float64)   # (N, 7, 7)

    @property
    def n_tracks(self):
        return len(self.xs)

    def add_track(self, z):
        """Add a new track from measurement z (4,) = [cx, cy, s, r]."""
        x_new = np.zeros(7, dtype=np.float64)
        x_new[:4] = z

        P_new = np.eye(7, dtype=np.float64) * 10.
        P_new[4:, 4:] *= 1000.  # high uncertainty on initial velocities

        if self.n_tracks == 0:
            self.xs = x_new[None, :]
            self.Ps = P_new[None, :, :]
        else:
            self.xs = np.vstack([self.xs, x_new[None, :]])
            self.Ps = np.concatenate([self.Ps, P_new[None, :, :]], axis=0)

    def remove_tracks(self, indices):
        """Remove tracks at given indices."""
        if len(indices) == 0:
            return
        keep = np.ones(self.n_tracks, dtype=bool)
        keep[indices] = False
        self.xs = self.xs[keep]
        self.Ps = self.Ps[keep]

    def predict_all(self):
        """Batched Kalman predict for all tracks at once."""
        if self.n_tracks == 0:
            return

        # Clamp: if scale + scale_velocity <= 0, zero out scale velocity
        mask = (self.xs[:, 2] + self.xs[:, 6]) <= 0
        self.xs[mask, 6] = 0.0

        # x = F @ x  →  (N,7) @ (7,7) = (N,7)
        self.xs = self.xs @ self.F_T

        # P = F @ P @ F.T + Q  →  broadcasts (7,7) @ (N,7,7) @ (7,7) + (7,7)
        self.Ps = self.F @ self.Ps @ self.F_T + self.Q

    def update_batch(self, track_indices, measurements):
        """
        Batched Kalman update for matched tracks only.
          track_indices: (M,) int array — which tracks were matched
          measurements:  (M, 4) float array — [cx, cy, s, r] observations
        """
        if len(track_indices) == 0:
            return

        idx = track_indices
        H = self.H         # (4, 7)
        H_T = self.H_T     # (7, 4)
        R = self.R          # (4, 4)

        x = self.xs[idx]    # (M, 7)
        P = self.Ps[idx]    # (M, 7, 7)
        z = measurements    # (M, 4)

        # Innovation: y = z - H @ x
        Hx = (H @ x[:, :, None]).squeeze(-1)  # (M, 4)
        y = z - Hx                             # (M, 4)

        # Innovation covariance: S = H @ P @ H.T + R
        S = H @ P @ H_T + R  # (M, 4, 4)

        # Kalman gain: K = P @ H.T @ inv(S)
        PHt = P @ H_T                     # (M, 7, 4)
        K = PHt @ np.linalg.inv(S)        # (M, 7, 4)

        # State update
        self.xs[idx] = x + (K @ y[:, :, None]).squeeze(-1)

        # Covariance update: P = (I - K @ H) @ P
        I_KH = self.I7 - K @ H  # (M, 7, 7)
        self.Ps[idx] = I_KH @ P

    def get_bboxes(self):
        """Return current state as bounding boxes (N, 4) = [x1, y1, x2, y2]."""
        return z_to_bbox(self.xs[:, :4])


# ---------- Main SORT tracker ----------

class FastSort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0

        # Batched Kalman state
        self.kf = BatchKalmanState()

        # Per-track metadata (parallel arrays, same indexing as kf.xs)
        self.time_since_update = np.empty(0, dtype=np.int32)
        self.hit_streak = np.empty(0, dtype=np.int32)
        self.hits = np.empty(0, dtype=np.int32)
        self.age = np.empty(0, dtype=np.int32)
        self.ids = np.empty(0, dtype=np.int32)

        self._next_id = 0

    def _add_track(self, bbox):
        """Create a new track from detection bbox [x1,y1,x2,y2]."""
        z = bbox_to_z(bbox[:4])[0]  # (4,)
        self.kf.add_track(z)
        self.time_since_update = np.append(self.time_since_update, 0)
        self.hit_streak = np.append(self.hit_streak, 0)
        self.hits = np.append(self.hits, 0)
        self.age = np.append(self.age, 0)
        self.ids = np.append(self.ids, self._next_id)
        self._next_id += 1

    def _remove_tracks(self, indices):
        """Remove tracks at given indices from all arrays."""
        if len(indices) == 0:
            return
        self.kf.remove_tracks(indices)
        keep = np.ones(len(self.time_since_update), dtype=bool)
        keep[indices] = False
        self.time_since_update = self.time_since_update[keep]
        self.hit_streak = self.hit_streak[keep]
        self.hits = self.hits[keep]
        self.age = self.age[keep]
        self.ids = self.ids[keep]

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - numpy array of detections [[x1,y1,x2,y2,score], ...]
        Returns:
          numpy array [[x1,y1,x2,y2,id], ...] for confirmed tracks.
        """
        self.frame_count += 1

        # --- Predict all tracks at once ---
        self.kf.predict_all()
        self.age += 1
        # Tracks not updated this frame lose their hit streak
        stale = self.time_since_update > 0
        self.hit_streak[stale] = 0
        self.time_since_update += 1

        # Get predicted bboxes, remove any with NaN
        if self.kf.n_tracks > 0:
            trk_bboxes = self.kf.get_bboxes()  # (N, 4)
            nan_mask = np.any(np.isnan(trk_bboxes), axis=1)
            if np.any(nan_mask):
                self._remove_tracks(np.where(nan_mask)[0])
                trk_bboxes = self.kf.get_bboxes()
        else:
            trk_bboxes = np.empty((0, 4))

        # --- Associate detections to tracks ---
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4] if len(dets) > 0 else np.empty((0, 4)),
            trk_bboxes,
            self.iou_threshold
        )

        # --- Update matched tracks (batched) ---
        if len(matched) > 0:
            trk_idx = matched[:, 1]
            det_idx = matched[:, 0]
            measurements = bbox_to_z(dets[det_idx, :4])  # (M, 4)
            self.kf.update_batch(trk_idx, measurements)
            self.time_since_update[trk_idx] = 0
            self.hits[trk_idx] += 1
            self.hit_streak[trk_idx] += 1

        # --- Create new tracks for unmatched detections ---
        for i in unmatched_dets:
            self._add_track(dets[i, :])

        # --- Build output and remove dead tracks ---
        # Remove tracks that have been lost for too long
        dead = np.where(self.time_since_update > self.max_age)[0]
        self._remove_tracks(dead)

        # Return confirmed tracks
        if self.kf.n_tracks == 0:
            return np.empty((0, 5))

        confirmed = ((self.time_since_update < 1) &
                      ((self.hit_streak >= self.min_hits) | (self.frame_count <= self.min_hits)))

        if not np.any(confirmed):
            return np.empty((0, 5))

        bboxes = self.kf.get_bboxes()[confirmed]       # (K, 4)
        track_ids = self.ids[confirmed] + 1             # +1 for MOT format
        return np.column_stack([bboxes, track_ids])     # (K, 5)