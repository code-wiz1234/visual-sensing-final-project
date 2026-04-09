"""Accumulate DVS events into images and detect objects via connected components."""

from __future__ import annotations

import cv2
import numpy as np


class EventAccumulator:
    """Accumulates events into 2D images over fixed time windows."""

    def __init__(self, height: int, width: int, window_sec: float = 0.01):
        self.height = height
        self.width = width
        self.window_sec = window_sec
        self.img = np.zeros((height, width), dtype=np.int16)
        self.window_start: float | None = None

    def add(self, t_sec: float, xs: np.ndarray, ys: np.ndarray, pols: np.ndarray):
        if len(xs) == 0:
            return
        in_bounds = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        xs = xs[in_bounds]; ys = ys[in_bounds]; pols = pols[in_bounds]
        np.add.at(self.img, (ys, xs), pols.astype(np.int16))

    def ready(self, t_sec: float) -> bool:
        if self.window_start is None:
            self.window_start = t_sec
            return False
        return (t_sec - self.window_start) >= self.window_sec

    def flush(self) -> np.ndarray:
        img = self.img
        self.img = np.zeros_like(img)
        self.window_start = None
        return img


class EventDetector:
    """Connected-components detector over an absolute-polarity event image."""

    def __init__(self, min_events_per_px: int = 1, min_area: int = 200,
                 max_area: int = 40000, dilate_iters: int = 2):
        self.min_events_per_px = min_events_per_px
        self.min_area = min_area
        self.max_area = max_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.dilate_iters = dilate_iters

    def __call__(self, event_img: np.ndarray) -> list[tuple[float, float, float, float]]:
        mask = (np.abs(event_img) >= self.min_events_per_px).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilate_iters)

        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for label_idx in range(1, num):  # skip background label 0
            x, y, w, h, area = stats[label_idx]
            if area < self.min_area or area > self.max_area:
                continue
            if h < 1.2 * w:  # skip wide/squat blobs, we're tracking people
                continue
            boxes.append((float(x), float(y), float(w), float(h)))
        return boxes
