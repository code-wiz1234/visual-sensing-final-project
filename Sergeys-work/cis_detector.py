"""Motion-based object detectors for intensity frames and event images."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def _mask_to_boxes(mask: np.ndarray, kernel, min_area: int, max_area: int,
                   aspect_filter: bool = True) -> list[tuple[float, float, float, float]]:
    """Clean up a foreground mask and return bounding boxes."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if aspect_filter and h < 1.2 * w:
            continue
        boxes.append((float(x), float(y), float(w), float(h)))
    return boxes


def _to_gray(frame_bgr, grayscale_flag):
    if grayscale_flag:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray
    return frame_bgr, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)



class FrameDiffDetector:
    """Simplest possible motion detector: threshold on |frame_t − frame_t-1|.

    This is the absolute baseline. If even this can detect objects on DVS
    event images, it shows DVS gives you motion for "free".
    """

    def __init__(self, threshold: int = 30, min_area: int = 400,
                 max_area: int = 40000, grayscale: bool = False):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.grayscale = grayscale
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.prev_gray: np.ndarray | None = None

    def __call__(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        _, gray = _to_gray(frame_bgr, self.grayscale)
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        return _mask_to_boxes(mask, self.kernel, self.min_area, self.max_area)



class MOG2Detector:
    """Background-subtraction detector over BGR or grayscale frames.

    Returns bboxes as a list of (x, y, w, h) tuples.
    """

    def __init__(self, min_area: int = 400, max_area: int = 40000,
                 history: int = 200, var_threshold: float = 25.0,
                 grayscale: bool = False):
        self.bgs = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        self.grayscale = grayscale
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __call__(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        frame, _ = _to_gray(frame_bgr, self.grayscale)
        mask = self.bgs.apply(frame)
        return _mask_to_boxes(mask, self.kernel, self.min_area, self.max_area)


class KNNDetector:
    """K-nearest-neighbors background subtractor (OpenCV built-in).

    Alternative to MOG2 — uses a sample-based KNN model instead of a
    parametric mixture-of-Gaussians. Often better on dynamic backgrounds.
    """

    def __init__(self, min_area: int = 400, max_area: int = 40000,
                 history: int = 200, dist2_threshold: float = 400.0,
                 grayscale: bool = False):
        self.bgs = cv2.createBackgroundSubtractorKNN(
            history=history, dist2Threshold=dist2_threshold, detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        self.grayscale = grayscale
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __call__(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        frame, _ = _to_gray(frame_bgr, self.grayscale)
        mask = self.bgs.apply(frame)
        return _mask_to_boxes(mask, self.kernel, self.min_area, self.max_area)


class OpticalFlowDetector:
    """Farneback dense optical-flow magnitude thresholding.

    Fundamentally different from bg-subtraction — detects *motion patterns*
    between consecutive frames by computing per-pixel flow vectors. Pixels
    whose flow magnitude exceeds `mag_threshold` are foreground.

    Should inherently favor DVS since event images ARE motion.
    """

    def __init__(self, mag_threshold: float = 2.0, min_area: int = 400,
                 max_area: int = 40000, grayscale: bool = False):
        self.mag_threshold = mag_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.grayscale = grayscale
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.prev_gray: np.ndarray | None = None

    def __call__(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        _, gray = _to_gray(frame_bgr, self.grayscale)
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mask = (mag > self.mag_threshold).astype(np.uint8) * 255
        return _mask_to_boxes(mask, self.kernel, self.min_area, self.max_area)



class PublicDetections:
    """Reads MOT17 det/det.txt and returns bboxes per frame."""

    def __init__(self, det_df: pd.DataFrame, min_conf: float = 0.3):
        self.by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
        for _, r in det_df.iterrows():
            if r["conf"] < min_conf:
                continue
            self.by_frame.setdefault(int(r["frame"]), []).append(
                (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            )

    def __call__(self, frame_idx: int, _frame_bgr=None) -> list[tuple[float, float, float, float]]:
        return self.by_frame.get(frame_idx, [])
