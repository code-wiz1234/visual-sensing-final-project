"""Centroid and IoU trackers -- lightweight alternatives to SORT.

Same update() API so the benchmark can swap them with one flag.
"""

from __future__ import annotations

import numpy as np


def _iou(bbox_a, bbox_b):
    x1 = max(bbox_a[0], bbox_b[0]); y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2]); y2 = min(bbox_a[3], bbox_b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _xywh_to_xyxy(bbox):
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


def _center(bbox_xywh):
    return (bbox_xywh[0] + bbox_xywh[2] / 2.0, bbox_xywh[1] + bbox_xywh[3] / 2.0)


class CentroidTracker:
    """Greedy nearest-centroid association. No motion model.

    A detection is matched to the nearest existing track (by bbox-center
    Euclidean distance) if that distance is below `max_distance`. Otherwise
    a new track is spawned. Tracks not seen for `max_age` frames are dropped.
    """

    def __init__(self, max_distance: float = 80.0, max_age: int = 5,
                 min_hits: int = 3):
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: dict[int, dict] = {}  # id -> {bbox, last_seen, hits}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections_xywh, histograms=None):
        self.frame_count += 1
        for track_id in self.tracks:
            self.tracks[track_id]["last_seen"] += 1

        if len(detections_xywh) == 0:
            self._prune()
            return self._emit()

        det_centers = np.array([_center(det) for det in detections_xywh])
        track_ids = list(self.tracks.keys())
        trk_centers = np.array([_center(self.tracks[track_id]["bbox"]) for track_id in track_ids]) \
            if track_ids else np.empty((0, 2))

        matched_dets: set[int] = set()
        matched_trks: set[int] = set()

        if track_ids:
            center_diffs = det_centers[:, None, :] - trk_centers[None, :, :]  # (N, M, 2)
            dist_mat = np.sqrt((center_diffs ** 2).sum(axis=2))  # (N, M)
            valid_pairs = np.argwhere(dist_mat <= self.max_distance)  # (K, 2)
            if len(valid_pairs):
                dists = dist_mat[valid_pairs[:, 0], valid_pairs[:, 1]]
                order = np.argsort(dists)
                for idx in order:
                    det_idx, track_idx = int(valid_pairs[idx, 0]), int(valid_pairs[idx, 1])
                    if det_idx in matched_dets or track_idx in matched_trks:
                        continue
                    matched_dets.add(det_idx)
                    matched_trks.add(track_idx)
                    track_id = track_ids[track_idx]
                    self.tracks[track_id]["bbox"] = detections_xywh[det_idx]
                    self.tracks[track_id]["last_seen"] = 0
                    self.tracks[track_id]["hits"] += 1

        for det_idx, det in enumerate(detections_xywh):
            if det_idx in matched_dets:
                continue
            self.tracks[self.next_id] = {"bbox": det, "last_seen": 0, "hits": 1}
            self.next_id += 1

        self._prune()
        return self._emit()

    def _prune(self):
        self.tracks = {track_id: track_state for track_id, track_state in self.tracks.items()
                       if track_state["last_seen"] <= self.max_age}

    def _emit(self):
        out = []
        for track_id, track_state in self.tracks.items():
            if track_state["last_seen"] == 0 and (track_state["hits"] >= self.min_hits
                                                   or self.frame_count <= self.min_hits):
                x, y, w, h = track_state["bbox"]
                out.append((track_id, float(x), float(y), float(w), float(h)))
        return out


class IoUTracker:
    """Greedy IoU association (Bochinski 2017 'high-speed tracking').

    For each detection, attach it to the existing track with which it has the
    highest IoU, provided IoU >= iou_threshold. Spawn new tracks for
    unmatched detections. Drop tracks not seen for max_age frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5,
                 min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections_xywh, histograms=None):
        self.frame_count += 1
        for track_id in self.tracks:
            self.tracks[track_id]["last_seen"] += 1

        if len(detections_xywh) == 0:
            self._prune()
            return self._emit()

        det_array = np.asarray(detections_xywh, dtype=float)
        det_xyxy = np.column_stack([det_array[:, 0], det_array[:, 1],
                                    det_array[:, 0]+det_array[:, 2],
                                    det_array[:, 1]+det_array[:, 3]])
        track_ids = list(self.tracks.keys())

        matched_dets: set[int] = set()
        matched_trks: set[int] = set()

        if track_ids:
            trk_boxes = np.array([_xywh_to_xyxy(self.tracks[track_id]["bbox"]) for track_id in track_ids])
            x1 = np.maximum(det_xyxy[:, 0:1], trk_boxes[:, 0:1].T)
            y1 = np.maximum(det_xyxy[:, 1:2], trk_boxes[:, 1:2].T)
            x2 = np.minimum(det_xyxy[:, 2:3], trk_boxes[:, 2:3].T)
            y2 = np.minimum(det_xyxy[:, 3:4], trk_boxes[:, 3:4].T)
            inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
            area_d = (det_xyxy[:, 2]-det_xyxy[:, 0]) * (det_xyxy[:, 3]-det_xyxy[:, 1])
            area_t = (trk_boxes[:, 2]-trk_boxes[:, 0]) * (trk_boxes[:, 3]-trk_boxes[:, 1])
            iou_mat = inter / (area_d[:, None] + area_t[None, :] - inter + 1e-9)

            valid = np.argwhere(iou_mat >= self.iou_threshold)
            if len(valid):
                ious = iou_mat[valid[:, 0], valid[:, 1]]
                order = np.argsort(-ious)  # descending IoU
                for idx in order:
                    det_idx, track_idx = int(valid[idx, 0]), int(valid[idx, 1])
                    if det_idx in matched_dets or track_idx in matched_trks:
                        continue
                    matched_dets.add(det_idx)
                    matched_trks.add(track_idx)
                    track_id = track_ids[track_idx]
                    self.tracks[track_id]["bbox"] = detections_xywh[det_idx]
                    self.tracks[track_id]["last_seen"] = 0
                    self.tracks[track_id]["hits"] += 1

        for det_idx, det in enumerate(detections_xywh):
            if det_idx in matched_dets:
                continue
            self.tracks[self.next_id] = {"bbox": det, "last_seen": 0, "hits": 1}
            self.next_id += 1

        self._prune()
        return self._emit()

    def _prune(self):
        self.tracks = {track_id: track_state for track_id, track_state in self.tracks.items()
                       if track_state["last_seen"] <= self.max_age}

    def _emit(self):
        out = []
        for track_id, track_state in self.tracks.items():
            if track_state["last_seen"] == 0 and (track_state["hits"] >= self.min_hits
                                                   or self.frame_count <= self.min_hits):
                x, y, w, h = track_state["bbox"]
                out.append((track_id, float(x), float(y), float(w), float(h)))
        return out
