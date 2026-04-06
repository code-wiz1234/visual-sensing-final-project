"""Convert RGB frames to DVS events. Uses v2e if installed, else a frame-diff fallback."""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

_V2E_AVAILABLE = False
try:
    from v2ecore.emulator import EventEmulator  # type: ignore
    _V2E_AVAILABLE = True
except Exception:
    pass


class _FrameDiffEmulator:
    """Log-intensity frame-diff emulator. Events fire where
    |log(I_t+eps) - log(I_ref+eps)| >= theta, polarity from sign.
    ref is updated to current I at pixels that fired."""

    def __init__(self, theta: float, epsilon: float = 1e-3):
        self.theta = theta
        self.epsilon = epsilon
        self.log_ref: np.ndarray | None = None

    def __call__(self, gray01: np.ndarray, t_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_intensity = np.log(gray01 + self.epsilon)
        if self.log_ref is None:
            self.log_ref = log_intensity.copy()
            return (np.empty(0, int), np.empty(0, int), np.empty(0, np.int8))
        diff = log_intensity - self.log_ref
        fired = np.abs(diff) >= self.theta
        ys, xs = np.where(fired)
        pol = np.sign(diff[ys, xs]).astype(np.int8)
        self.log_ref[ys, xs] = log_intensity[ys, xs]
        return xs.astype(int), ys.astype(int), pol


def generate_events(
    frames: Iterable[tuple[int, np.ndarray]],
    frame_rate: float,
    theta: float,
    use_v2e: bool = True,
) -> Iterator[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (t_sec, xs, ys, pols) per frame. Arrays may be empty.

    Args:
        frames: iterable of (frame_idx, BGR_uint8) from ingest_mot.iter_frames.
        frame_rate: FPS of the source video.
        theta: contrast threshold (log-intensity). Matches dvs_model.BASELINE_THETA.
        use_v2e: try the real v2e emulator if installed; else use the frame-diff fallback.
    """
    import cv2

    dt = 1.0 / frame_rate
    backend = None

    if use_v2e and _V2E_AVAILABLE:
        backend = EventEmulator(
            pos_thres=theta,
            neg_thres=theta,
            sigma_thres=0.03,
            cutoff_hz=200,
            leak_rate_hz=0.1,
            shot_noise_rate_hz=1.0,
            output_folder=None,
            dvs_h5=None,
            dvs_aedat2=None,
            dvs_text=None,
        )
    else:
        backend = _FrameDiffEmulator(theta=theta)

    for frame_idx, bgr in frames:
        t_sec = (frame_idx - 1) * dt
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        if use_v2e and _V2E_AVAILABLE:
            ev = backend.generate_events(gray * 255.0, t_sec)
            if ev is None or len(ev) == 0:
                yield t_sec, np.empty(0, int), np.empty(0, int), np.empty(0, np.int8)
                continue
            # v2e output columns: [t, x, y, p] with p in {-1, 1}
            timestamps = ev[:, 0]
            x_coords = ev[:, 1].astype(int)
            y_coords = ev[:, 2].astype(int)
            polarities = ev[:, 3].astype(np.int8)
            yield float(timestamps.mean()) if len(timestamps) else t_sec, x_coords, y_coords, polarities
        else:
            xs, ys, ps = backend(gray, t_sec)
            yield t_sec, xs, ys, ps


def is_v2e_available() -> bool:
    return _V2E_AVAILABLE
