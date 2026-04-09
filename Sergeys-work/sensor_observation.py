"""Simulate CIS and DVS sensor observations from ground-truth trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _velocity_per_object(gt_df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Compute per-object, per-frame velocity from GT displacement."""
    gt = gt_df.sort_values(["id", "frame"]).copy()
    gt["cx"] = gt["x"] + gt["w"] / 2
    gt["cy"] = gt["y"] + gt["h"] / 2
    gt["prev_cx"] = gt.groupby("id")["cx"].shift(1)
    gt["prev_cy"] = gt.groupby("id")["cy"].shift(1)
    dx = gt["cx"] - gt["prev_cx"]
    dy = gt["cy"] - gt["prev_cy"]
    gt["displacement_px"] = np.sqrt(dx**2 + dy**2).fillna(0.0)
    gt["velocity_px_s"] = (gt["displacement_px"] * fps).fillna(0.0)
    return gt



def simulate_cis(gt_df: pd.DataFrame, fps: float,
                 actual_fps: float | None = None,
                 resolution: tuple[int, int] | None = None,
                 adc_bits: int = 12,
                 velocity_scale: float = 1.0,
                 exposure_fraction: float = 0.5,
                 rng: np.random.Generator | None = None) -> pd.DataFrame:
    """CIS simulation -- models FPS subsampling, resolution, ADC noise, and motion blur."""
    if rng is None:
        rng = np.random.default_rng(42)
    if actual_fps is None:
        actual_fps = fps

    REFERENCE_WIDTH = 1920
    sensor_width = resolution[0] if resolution else REFERENCE_WIDTH
    spatial_scale = REFERENCE_WIDTH / sensor_width
    resolution_noise_sigma = 0.5 * spatial_scale

    adc_noise_factor = 2.0 ** (12 - adc_bits)
    adc_noise_sigma = 0.3 * adc_noise_factor
    adc_miss_prob = 0.02 * (adc_noise_factor - 1)

    gt = _velocity_per_object(gt_df, fps)
    gt["velocity_px_s"] = gt["velocity_px_s"] * velocity_scale
    gt["displacement_px"] = gt["displacement_px"] * velocity_scale

    all_frames = sorted(gt["frame"].unique())
    if actual_fps < fps:
        frame_step = max(1, int(round(fps / actual_fps)))
        observed_frames = set(all_frames[::frame_step])
    else:
        observed_frames = set(all_frames)

    effective_fps = min(actual_fps, fps)
    sensor_dt = 1.0 / effective_fps

    obs = gt[gt["frame"].isin(observed_frames)].copy()
    if obs.empty:
        return pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h"])

    num_obs = len(obs)
    velocities = obs["velocity_px_s"].values
    widths = obs["w"].values
    heights = obs["h"].values

    # displacement miss: object moves > 2x its width between CIS frames
    disp = velocities * sensor_dt
    keep = disp <= 2.0 * widths

    if adc_miss_prob > 0:
        keep &= rng.random(num_obs) >= adc_miss_prob

    obs = obs[keep].copy()
    if obs.empty:
        return pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h"])

    num_obs = len(obs)
    velocities = obs["velocity_px_s"].values
    widths = obs["w"].values
    heights = obs["h"].values

    blur_sigma = velocities * (exposure_fraction * sensor_dt)
    total_pos_sigma = np.sqrt(blur_sigma**2 + resolution_noise_sigma**2 + adc_noise_sigma**2)
    total_pos_sigma = np.maximum(total_pos_sigma, 0.3)

    noise_x = rng.normal(0, total_pos_sigma)
    noise_y = rng.normal(0, total_pos_sigma)

    size_sigma = 0.03 + 0.02 * (spatial_scale - 1)
    noise_w = rng.normal(0, size_sigma * widths)
    noise_h = rng.normal(0, size_sigma * heights)

    result = pd.DataFrame({
        "frame": obs["frame"].values,
        "id": obs["id"].values,
        "x": obs["x"].values + noise_x,
        "y": obs["y"].values + noise_y,
        "w": np.maximum(1, widths + noise_w),
        "h": np.maximum(1, heights + noise_h),
    })

    # If actual_fps > source fps, multiple sensor samples map to same frame.
    # Average them (higher fps = better position estimate).
    if actual_fps > fps:
        result = result.groupby(["frame", "id"], as_index=False).mean()

    return result



def simulate_dvs(gt_df: pd.DataFrame, fps: float,
                 pixel_latency_s: float = 15e-6,
                 refractory_cap: float = 18.75e6,
                 resolution: tuple[int, int] | None = None,
                 contrast_threshold: float = 0.10,
                 velocity_scale: float = 1.0,
                 min_velocity_px_s: float = 5.0,
                 coast_frames: int = 150,
                 rng: np.random.Generator | None = None) -> pd.DataFrame:
    """DVS simulation -- models event detection, refractory saturation, and coasting."""
    if rng is None:
        rng = np.random.default_rng(42)

    BASELINE_THETA = 0.05
    theta_scale = BASELINE_THETA / contrast_threshold
    MIN_EVENTS_PER_FRAME = 50

    REFERENCE_WIDTH = 1920
    sensor_width = resolution[0] if resolution else 640
    spatial_scale = REFERENCE_WIDTH / sensor_width
    resolution_noise_sigma = 0.5 * spatial_scale

    latency_factor = np.sqrt(pixel_latency_s / 1e-6) * 0.05
    size_scale = latency_factor + 0.02 * (spatial_scale - 1)

    gt = _velocity_per_object(gt_df, fps)
    gt["velocity_px_s"] = gt["velocity_px_s"] * velocity_scale
    gt["displacement_px"] = gt["displacement_px"] * velocity_scale

    num_rows = len(gt)
    velocities = gt["velocity_px_s"].values
    widths = gt["w"].values
    heights = gt["h"].values

    is_moving = velocities >= min_velocity_px_s

    perimeter = 2 * (widths + heights)
    event_rate = perimeter * velocities * theta_scale
    effective_event_rate = np.minimum(event_rate, refractory_cap)

    sat_miss_prob = np.where(event_rate > refractory_cap,
                             1.0 - refractory_cap / np.maximum(event_rate, 1),
                             0.0)
    sat_pass = rng.random(num_rows) >= sat_miss_prob

    events_per_frame = effective_event_rate / fps
    sparse_miss_prob = np.where(events_per_frame < MIN_EVENTS_PER_FRAME,
                                1.0 - events_per_frame / MIN_EVENTS_PER_FRAME,
                                0.0)
    sparse_pass = rng.random(num_rows) >= sparse_miss_prob

    detected = is_moving & sat_pass & sparse_pass

    latency_sigma = velocities * pixel_latency_s
    pos_sigma = np.sqrt(latency_sigma**2 + resolution_noise_sigma**2)
    pos_sigma = np.maximum(pos_sigma, 0.1)

    noise_x = rng.normal(0, pos_sigma)
    noise_y = rng.normal(0, pos_sigma)
    noise_w = rng.normal(0, size_scale * widths)
    noise_h = rng.normal(0, size_scale * heights)

    det_x = gt["x"].values + noise_x
    det_y = gt["y"].values + noise_y
    det_w = np.maximum(1, widths + noise_w)
    det_h = np.maximum(1, heights + noise_h)

    frame_numbers = gt["frame"].values
    object_ids = gt["id"].values
    unique_frames = sorted(gt["frame"].unique())

    out_frame = []
    out_id = []
    out_x = []
    out_y = []
    out_w = []
    out_h = []

    last_bbox = {}              # object_id -> (x, y, w, h)
    frames_since_detection = {} # object_id -> frames since last event detection

    for frame in unique_frames:
        mask = frame_numbers == frame
        frame_idx = np.where(mask)[0]
        active_ids = set()

        for row_idx in frame_idx:
            object_id = int(object_ids[row_idx])
            active_ids.add(object_id)

            if detected[row_idx]:
                box_x, box_y, box_w, box_h = det_x[row_idx], det_y[row_idx], det_w[row_idx], det_h[row_idx]
                out_frame.append(frame)
                out_id.append(object_id)
                out_x.append(box_x)
                out_y.append(box_y)
                out_w.append(box_w)
                out_h.append(box_h)
                last_bbox[object_id] = (box_x, box_y, box_w, box_h)
                frames_since_detection[object_id] = 0
            elif object_id in last_bbox and frames_since_detection.get(object_id, coast_frames) < coast_frames:
                # Coast: use last known position with drift
                box_x, box_y, box_w, box_h = last_bbox[object_id]
                coast_age = frames_since_detection[object_id] + 1
                drift = coast_age * 0.3
                out_frame.append(frame)
                out_id.append(object_id)
                out_x.append(box_x + rng.normal(0, drift))
                out_y.append(box_y + rng.normal(0, drift))
                out_w.append(box_w)
                out_h.append(box_h)
                frames_since_detection[object_id] = coast_age

        # Remove objects that left the scene
        for object_id in list(last_bbox.keys()):
            if object_id not in active_ids:
                del last_bbox[object_id]
                frames_since_detection.pop(object_id, None)

    if not out_frame:
        return pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h"])

    return pd.DataFrame({
        "frame": out_frame, "id": out_id,
        "x": out_x, "y": out_y, "w": out_w, "h": out_h,
    })
