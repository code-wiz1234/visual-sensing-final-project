"""Fast SORT tracker -- same algorithm as sort_tracker.py, no filterpy dependency."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_batch(detections: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy bounding boxes."""
    x1 = np.maximum(detections[:, 0:1], tracks[:, 0:1].T)
    y1 = np.maximum(detections[:, 1:2], tracks[:, 1:2].T)
    x2 = np.minimum(detections[:, 2:3], tracks[:, 2:3].T)
    y2 = np.minimum(detections[:, 3:4], tracks[:, 3:4].T)
    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih
    area_d = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
    area_t = (tracks[:, 2] - tracks[:, 0]) * (tracks[:, 3] - tracks[:, 1])
    return inter / (area_d[:, None] + area_t[None, :] - inter + 1e-9)


# Pre-built constant matrices for the 7-state Kalman filter
# State: [cx, cy, s, r, vx, vy, vs] where s=area, r=aspect ratio
_F = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
], dtype=np.float64)

_H = np.eye(4, 7, dtype=np.float64)

# Process noise
_Q = np.eye(7, dtype=np.float64)
_Q[4:, 4:] *= 0.01
_Q[6, 6] *= 0.01

# Measurement noise
_R = np.eye(4, dtype=np.float64)
_R[2:, 2:] *= 10.0

# Initial covariance
def _init_P():
    P = np.eye(7, dtype=np.float64) * 10.0
    P[4:, 4:] *= 1000.0
    return P


def _xyxy_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([bbox[0] + w/2, bbox[1] + h/2, w*h, w/(h+1e-9)], dtype=np.float64)


def _z_to_xyxy(z):
    w = np.sqrt(max(z[2] * z[3], 1e-9))
    h = z[2] / (w + 1e-9)
    return np.array([z[0] - w/2, z[1] - h/2, z[0] + w/2, z[1] + h/2])


class _Track:
    _next_id = 1

    def __init__(self, bbox_xyxy):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.x = np.zeros(7, dtype=np.float64)
        self.x[:4] = _xyxy_to_z(bbox_xyxy)
        self.P = _init_P()
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy):
        z = _xyxy_to_z(bbox_xyxy)
        y = z - _H @ self.x                          # innovation
        S = _H @ self.P @ _H.T + _R                  # innovation covariance
        K = self.P @ _H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ _H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    def bbox(self):
        return _z_to_xyxy(self.x[:4])


class Sort:
    """Drop-in replacement for sort_tracker.Sort with same API."""

    def __init__(self, max_age: int = 5, min_hits: int = 3,
                 iou_threshold: float = 0.3, color_gate: float = 0.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[_Track] = []
        self.frame_count = 0

    def update(self, detections_xywh, histograms=None):
        self.frame_count += 1

        for track in self.tracks:
            track.predict()
        self.tracks = [track for track in self.tracks
                       if not np.any(np.isnan(track.bbox()))]

        if len(detections_xywh):
            det_array = np.asarray(detections_xywh, dtype=np.float64)
            dets = np.column_stack([det_array[:, 0], det_array[:, 1],
                                    det_array[:, 0]+det_array[:, 2],
                                    det_array[:, 1]+det_array[:, 3]])
        else:
            dets = np.empty((0, 4))

        num_dets, num_tracks = len(dets), len(self.tracks)
        matches = []
        unmatched_dets = list(range(num_dets))

        if num_dets and num_tracks:
            trk_bboxes = np.array([track.bbox() for track in self.tracks])
            iou_mat = _iou_batch(dets, trk_bboxes)
            cost = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost)

            unmatched_dets = []
            unmatched_trks = list(range(num_tracks))
            for det_idx, trk_idx in zip(row_ind, col_ind):
                if iou_mat[det_idx, trk_idx] < self.iou_threshold:
                    unmatched_dets.append(det_idx)
                else:
                    matches.append((det_idx, trk_idx))
                    if trk_idx in unmatched_trks:
                        unmatched_trks.remove(trk_idx)
            for det_idx in range(num_dets):
                if det_idx not in [m[0] for m in matches] and det_idx not in unmatched_dets:
                    unmatched_dets.append(det_idx)

        for det_i, trk_i in matches:
            self.tracks[trk_i].update(dets[det_i])

        for det_idx in unmatched_dets:
            self.tracks.append(_Track(dets[det_idx]))

        self.tracks = [track for track in self.tracks
                       if track.time_since_update <= self.max_age]

        out = []
        for track in self.tracks:
            if track.time_since_update == 0 and (track.hits >= self.min_hits
                                                  or self.frame_count <= self.min_hits):
                x1, y1, x2, y2 = track.bbox()
                out.append((track.id, float(x1), float(y1), float(x2-x1), float(y2-y1)))
        return out
