"""SORT: Simple Online and Realtime Tracker.

Kalman filter (constant-velocity) per track + Hungarian IoU association.
Follows Bewley et al. 2016 closely. Optional color-histogram gating is
used by the CIS-RGB variant to break ties in crowded scenes.
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def _iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(bbox_a[0], bbox_b[0]); y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2]); y2 = min(bbox_a[3], bbox_b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _iou_batch(dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
    """IoU between detections and tracks (xyxy format)."""
    x1 = np.maximum(dets[:, 0:1], trks[:, 0:1].T)  # (N, M)
    y1 = np.maximum(dets[:, 1:2], trks[:, 1:2].T)
    x2 = np.minimum(dets[:, 2:3], trks[:, 2:3].T)
    y2 = np.minimum(dets[:, 3:4], trks[:, 3:4].T)
    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih
    area_d = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])  # (N,)
    area_t = (trks[:, 2] - trks[:, 0]) * (trks[:, 3] - trks[:, 1])  # (M,)
    union = area_d[:, None] + area_t[None, :] - inter + 1e-9
    return inter / union


def _xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=float)


def _xyxy_to_z(bbox):
    """xyxy -> [cx, cy, s, r] where s=area, r=aspect."""
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2; cy = bbox[1] + h / 2
    s = w * h
    r = w / (h + 1e-9)
    return np.array([cx, cy, s, r]).reshape(4, 1)


def _z_to_xyxy(z):
    cx, cy, s, r = z.flatten()[:4]
    w = np.sqrt(max(s * r, 1e-9))
    h = s / (w + 1e-9)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class _Track:
    _next_id = 1

    def __init__(self, bbox_xyxy: np.ndarray, hist: np.ndarray | None = None):
        self.id = _Track._next_id
        _Track._next_id += 1

        # State: [cx, cy, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.eye(4, 7)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # high uncertainty on velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _xyxy_to_z(bbox_xyxy)

        self.hist = hist
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def predict(self):
        # Guard against degenerate area
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy: np.ndarray, hist: np.ndarray | None = None):
        self.kf.update(_xyxy_to_z(bbox_xyxy))
        self.time_since_update = 0
        self.hits += 1
        if hist is not None:
            # EMA of color histogram for stable appearance
            self.hist = 0.7 * self.hist + 0.3 * hist if self.hist is not None else hist

    def bbox(self) -> np.ndarray:
        return _z_to_xyxy(self.kf.x)


class Sort:
    """Multi-object tracker.

    Args:
        max_age: drop tracks not updated for this many frames.
        min_hits: only emit tracks after this many consecutive hits.
        iou_threshold: minimum IoU for association.
        color_gate: if >0, reject associations whose color-hist distance
            exceeds this threshold (Bhattacharyya).
    """

    def __init__(self, max_age: int = 5, min_hits: int = 3,
                 iou_threshold: float = 0.3, color_gate: float = 0.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.color_gate = color_gate
        self.tracks: list[_Track] = []
        self.frame_count = 0

    def update(self, detections_xywh: np.ndarray,
               histograms: list[np.ndarray] | None = None) -> list[tuple[int, float, float, float, float]]:
        """Advance one frame. detections_xywh is (N,4) in xywh pixel coords.

        Returns list of (track_id, x, y, w, h) for confirmed tracks.
        """
        self.frame_count += 1

        for track in self.tracks:
            track.predict()

        self.tracks = [track for track in self.tracks
                       if not np.any(np.isnan(track.bbox()))]

        if len(detections_xywh):
            det_array = np.asarray(detections_xywh, dtype=float)
            dets_xyxy = np.column_stack([det_array[:, 0], det_array[:, 1],
                                         det_array[:, 0] + det_array[:, 2],
                                         det_array[:, 1] + det_array[:, 3]])
        else:
            dets_xyxy = np.empty((0, 4))

        num_dets, num_tracks = len(dets_xyxy), len(self.tracks)
        matches: list[tuple[int, int]] = []
        unmatched_dets = list(range(num_dets))
        unmatched_trks = list(range(num_tracks))

        if num_dets and num_tracks:
            trk_bboxes = np.array([track.bbox() for track in self.tracks])
            iou_mat = _iou_batch(dets_xyxy, trk_bboxes)
            cost = 1.0 - iou_mat

            if self.color_gate > 0.0 and histograms is not None:
                for det_idx in range(num_dets):
                    for trk_idx in range(num_tracks):
                        trk = self.tracks[trk_idx]
                        if trk.hist is None or histograms[det_idx] is None:
                            continue
                        color_dist = _bhattacharyya(histograms[det_idx], trk.hist)
                        if color_dist > self.color_gate:
                            cost[det_idx, trk_idx] = 1.0

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
            hist = histograms[det_i] if histograms is not None else None
            self.tracks[trk_i].update(dets_xyxy[det_i], hist)

        for det_idx in unmatched_dets:
            hist = histograms[det_idx] if histograms is not None else None
            self.tracks.append(_Track(dets_xyxy[det_idx], hist))

        self.tracks = [track for track in self.tracks
                       if track.time_since_update <= self.max_age]

        out = []
        for track in self.tracks:
            if track.time_since_update == 0 and (track.hits >= self.min_hits
                                                  or self.frame_count <= self.min_hits):
                x1, y1, x2, y2 = track.bbox()
                out.append((track.id, float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
        return out


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance between two L1-normalized histograms."""
    return float(np.sqrt(1.0 - np.sum(np.sqrt(h1 * h2))))


def color_histogram(frame_bgr: np.ndarray, bbox_xywh: tuple[float, float, float, float],
                    bins: int = 8) -> np.ndarray | None:
    """8-bin-per-channel HSV histogram over the bbox, L1-normalized."""
    x, y, w, h = bbox_xywh
    x0 = max(0, int(x)); y0 = max(0, int(y))
    x1 = min(frame_bgr.shape[1], int(x + w))
    y1 = min(frame_bgr.shape[0], int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = frame_bgr[y0:y1, x0:x1]
    hsv = cv2_cvtColor(roi)
    hist = _hsv_hist(hsv, bins)
    s = hist.sum()
    return hist / s if s > 0 else None


def cv2_cvtColor(roi):
    import cv2
    return cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


def _hsv_hist(hsv, bins):
    import cv2
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    return hist.flatten()
