"""DVS vs CIS tracking baseline -- synthetic scenes through the real tracker pipeline.

Generates synthetic object trajectories from scene parameters (velocity, size,
background), then runs them through the same sensor simulation -> tracker ->
MOTA evaluation pipeline as the real-world MOT17 benchmark.

Output matches the same CSV schema as run_crossover.py so the same slide
generator works for both synthetic and real-world data.

Author: Sergey Petrushkevich
EECE5698 - Visual Sensing and Computing
"""

import os
import sys
import time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
OUT_DIR = BASE

# import from real-world pipeline
sys.path.insert(0, BASE)
from sensor_database import DVS_SENSORS, CIS_SENSORS
from sensor_observation import simulate_dvs, simulate_cis
from fast_eval import evaluate, tracks_to_df
from fast_sort import Sort as FastSort
from simple_trackers import CentroidTracker, IoUTracker

# import scene model
sys.path.insert(0, os.path.join(PROJECT, "Remaas-work"))
from visualcomputing import (compute_event_rate, compute_fps_min,
                              velocities, backgrounds,
                              false_pos_rate, scene_width, scene_height)



def generate_synthetic_gt(velocity_px_s, obj_size_px, num_objects=5,
                          num_frames=300, fps=30.0):
    """Create synthetic GT trajectories -- objects moving at a given velocity.

    Returns a DataFrame in MOT17 GT format (frame, id, x, y, w, h)
    that feeds directly into simulate_dvs() / simulate_cis().
    """
    rows = []
    dt = 1.0 / fps

    for object_id in range(1, num_objects + 1):
        rng = np.random.default_rng(42 + object_id)
        start_y = rng.uniform(obj_size_px, scene_height - obj_size_px)
        start_x = rng.uniform(0, scene_width * 0.3)
        y_drift = rng.uniform(-0.1, 0.1) * velocity_px_s

        for frame_num in range(1, num_frames + 1):
            elapsed_time_s = (frame_num - 1) * dt
            current_x = (start_x + velocity_px_s * elapsed_time_s) % (scene_width + obj_size_px)
            current_y = np.clip(start_y + y_drift * elapsed_time_s, 0, scene_height - obj_size_px)

            rows.append({
                "frame": frame_num, "id": object_id,
                "x": current_x, "y": current_y,
                "w": float(obj_size_px), "h": float(obj_size_px * 2),
            })

    return pd.DataFrame(rows)


def make_tracker(kind="sort"):
    if kind == "centroid":
        return CentroidTracker(max_distance=80.0, max_age=5, min_hits=3)
    if kind == "iou":
        return IoUTracker(iou_threshold=0.3, max_age=5, min_hits=3)
    if kind == "sort":
        return FastSort(max_age=5, min_hits=3, iou_threshold=0.3)
    raise ValueError(f"unknown tracker: {kind}")


def _run_one(config):
    """Worker function for ProcessPoolExecutor -- runs one config end to end.

    Imports are repeated inside the function so each worker process has them,
    same pattern as run_crossover._run_one.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "Remaas-work"))

    from sensor_observation import simulate_dvs, simulate_cis
    from fast_eval import evaluate, tracks_to_df
    from fast_sort import Sort as FastSort
    from simple_trackers import CentroidTracker, IoUTracker
    from visualcomputing import compute_event_rate, compute_fps_min, false_pos_rate

    velocity = config["velocity"]
    object_size = config["obj_size"]
    background_density = config["bg_density"]
    sensor = config["sensor"]
    tracker_kind = config["tracker"]
    fps = config["fps"]
    num_frames = config["num_frames"]
    num_seeds = config["num_seeds"]
    contrast_threshold = config.get("contrast_threshold")
    actual_fps_override = config.get("actual_fps_override")

    gt_df = generate_synthetic_gt(velocity, object_size, num_frames=num_frames, fps=fps)
    event_rate = compute_event_rate(velocity, object_size, background_density, false_pos_rate)
    required_fps = compute_fps_min(velocity, object_size)

    sensor_config = {**sensor, "_velocity": vel}

    if sensor["type"] == "DVS":
        ct = contrast_threshold if contrast_threshold is not None else 0.10
        sensor_config["contrast_threshold"] = ct
        baseline_theta = 0.05
        theta_scale = baseline_theta / ct
        eff_rate = min(event_rate * theta_scale, sensor["refractory_cap"])
        power = sensor["p_static_mw"] + eff_rate * sensor["e_per_event_nj"] * 1e-9 * 1e3
        fps_fraction = None
        actual_fps = None
    else:
        ct = None
        if actual_fps_override is not None:
            actual_fps = actual_fps_override
        else:
            actual_fps = sensor["max_fps"]
        sensor_config["actual_fps"] = actual_fps
        frac = min(actual_fps / sensor["max_fps"], 1.0)
        power = sensor["power_idle_mw"] + frac * (sensor["power_max_mw"] - sensor["power_idle_mw"])
        fps_fraction = round(frac, 2)

    # run tracker pipeline, average over seeds
    empty_dets = np.empty((0, 4))
    all_metrics = []

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx + int(vel)
        rng = np.random.default_rng(seed if sensor["type"] == "DVS" else seed + 10000)

        if sensor_config["type"] == "DVS":
            noisy_dets = simulate_dvs(
                gt_df, fps=fps,
                pixel_latency_s=sensor_config["pixel_latency_s"],
                refractory_cap=sensor_config["refractory_cap"],
                resolution=sensor_config.get("resolution"),
                contrast_threshold=ct or 0.10,
                velocity_scale=1.0, rng=rng)
        else:
            noisy_dets = simulate_cis(
                gt_df, fps=fps,
                actual_fps=sensor_config.get("actual_fps"),
                resolution=sensor_config.get("resolution"),
                adc_bits=sensor_config.get("adc_bits", 12),
                velocity_scale=1.0, rng=rng)

        det_by_frame = {}
        if not noisy_dets.empty:
            for frame_key, frame_grp in noisy_dets.groupby("frame"):
                det_by_frame[int(frame_key)] = frame_grp[["x", "y", "w", "h"]].to_numpy()

        tracker = make_tracker(tracker_kind)
        tracks_per_frame = []
        for frame_num in range(1, num_frames + 1):
            dets = det_by_frame.get(frame_num, empty_dets)
            trks = tracker.update(dets)
            tracks_per_frame.append((frame_num, trks))

        pred_df = tracks_to_df(tracks_per_frame)
        all_metrics.append(evaluate(pred_df, gt_df))

    metrics = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if m[key] is not None]
        if vals and isinstance(vals[0], (int, float)):
            metrics[key] = sum(vals) / len(vals)
        else:
            metrics[key] = all_metrics[0][key]

    return {
        "sensor_name": sensor["name"],
        "sensor_type": sensor["type"],
        "tracker": tracker_kind,
        "velocity_scale": velocity,
        "mean_velocity_px_s": velocity,
        "power_mW": round(power, 2),
        "required_fps": round(required_fps, 1) if sensor["type"] == "CIS" else None,
        "actual_fps": actual_fps,
        "contrast_threshold": ct,
        "fps_fraction": fps_fraction,
        "price": sensor["price"],
        **metrics,
    }


def run_synthetic_crossover(tracker_kinds=None, num_seeds=3, max_workers=None):
    """Sweep all sensors x velocities with knob variations, parallelized.

    Matches the real-world run_crossover_data() structure:
      - Baseline: all sensors at default operating point
      - CIS FPS sweep: 5 and 15 fps (lower power, worse tracking)
      - DVS theta sweep: 0.01 and 0.03 (more sensitive, more power)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if tracker_kinds is None:
        tracker_kinds = ["sort"]
    if max_workers is None:
        max_workers = min(12, os.cpu_count() or 4)

    # build sensor configs
    dvs_configs = []
    for dvs in DVS_SENSORS:
        dvs_configs.append({
            "name": dvs.name, "type": "DVS",
            "pixel_latency_s": dvs.pixel_latency_us * 1e-6,
            "refractory_cap": dvs.max_event_rate_mevps * 1e6,
            "resolution": dvs.resolution,
            "p_static_mw": dvs.p_static_mw,
            "e_per_event_nj": dvs.e_per_event_nj,
            "price": dvs.price_usd,
        })
    cis_configs = []
    for cis in CIS_SENSORS:
        cis_configs.append({
            "name": cis.name, "type": "CIS",
            "max_fps": cis.max_fps,
            "resolution": cis.resolution,
            "adc_bits": cis.adc_bits,
            "actual_fps": cis.max_fps,
            "power_idle_mw": cis.power_idle_mw,
            "power_max_mw": cis.power_at_max_fps_mw,
            "price": cis.price_usd,
        })

    object_size = 50
    background_density = backgrounds["low_texture"]
    source_fps = 30.0
    num_frames = 300
    all_sensors = dvs_configs + cis_configs
    cis_fps_settings = [5, 15]
    dvs_theta_settings = [0.01, 0.03]

    # build all configs as dicts (picklable for multiprocessing)
    configs = []
    base = {"obj_size": object_size, "bg_density": background_density,
            "fps": source_fps, "num_frames": num_frames, "num_seeds": num_seeds}

    # baseline: all sensors at default (theta=0.10, max FPS)
    for sensor in all_sensors:
        for tracker_kind in tracker_kinds:
            for vel in velocities:
                configs.append({**base, "sensor": sensor, "tracker": tracker_kind,
                                "velocity": vel})

    # CIS FPS sweep
    for sensor in cis_configs:
        for fps_setting in cis_fps_settings:
            for tracker_kind in tracker_kinds:
                for vel in velocities:
                    configs.append({**base, "sensor": sensor, "tracker": tracker_kind,
                                    "velocity": vel, "actual_fps_override": fps_setting})

    # DVS theta sweep
    for sensor in dvs_configs:
        for theta in dvs_theta_settings:
            for tracker_kind in tracker_kinds:
                for vel in velocities:
                    configs.append({**base, "sensor": sensor, "tracker": tracker_kind,
                                    "velocity": vel, "contrast_threshold": theta})

    num_configs = len(configs)
    print(f"Running {num_configs} configs on {max_workers} workers "
          f"({os.cpu_count()} CPU threads available)...\n")

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, config): config for config in configs}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                rows.append(result)
                if done % 40 == 0 or done == num_configs:
                    print(f"  [{done:3d}/{num_configs}] completed...")
            except Exception as exc:
                failed = futures[future]
                print(f"  [{done:3d}/{num_configs}] FAILED "
                      f"{failed['sensor']['name']} {failed['tracker']} "
                      f"v={failed['velocity']}: {exc}")

    return pd.DataFrame(rows)



def main():
    print("=" * 70)
    print("Synthetic Tracking Baseline")
    print("Same pipeline as real-world: sensor sim -> tracker -> MOTA")
    print("=" * 70)

    t_start = time.time()

    # run the sweep -- same schema as crossover_results.csv
    df = run_synthetic_crossover(tracker_kinds=["centroid", "iou", "sort"], num_seeds=3)

    out_path = os.path.join(OUT_DIR, "synthetic_crossover_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # generate slides using the same template as real-world
    # (already on sys.path from the top-level import)
    from generate_slides import generate_all_slides  # noqa: E402
    synthetic_slides_dir = os.path.join(OUT_DIR, "synthetic_slides")
    generate_all_slides(df, output_dir=synthetic_slides_dir,
                        source_label="Synthetic Scenes")

    wall = time.time() - t_start
    print(f"\nDone in {wall:.1f}s")
    print(f"Slides saved to: {synthetic_slides_dir}/")


if __name__ == "__main__":
    main()
