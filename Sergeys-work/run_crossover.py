"""Find the DVS/CIS tracking crossover by sweeping velocity across sensors and trackers."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MOT_ROOT = os.path.join(THIS_DIR, "MOT17")


def _run_one(config: dict) -> dict:
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Ishs-work")))

    from ingest_mot import load_seqinfo, load_gt
    from sensor_observation import simulate_cis, simulate_dvs
    from fast_eval import evaluate
    from run_benchmark import make_tracker

    seq_dir = os.path.join(config["mot_root"], "train", config["seq"])
    info = load_seqinfo(seq_dir)
    gt_df = load_gt(info)
    max_frames = config["max_frames"]
    gt_sub = gt_df[gt_df["frame"] <= max_frames].copy()

    sensor = config["sensor"]
    tracker_kind = config["tracker"]
    vel_scale = config["velocity_scale"]
    base_seed = config.get("seed", 42)
    seeds = [base_seed + vel_scale * 1000 + seed_idx for seed_idx in range(5)]

    from fast_eval import tracks_to_df
    all_metrics = []
    for seed in seeds:
        rng = np.random.default_rng(seed)

        if sensor["type"] == "DVS":
            noisy_dets = simulate_dvs(
                gt_sub, fps=info.frame_rate,
                pixel_latency_s=sensor["pixel_latency_s"],
                refractory_cap=sensor["refractory_cap"],
                resolution=sensor.get("resolution"),
                contrast_threshold=config.get("contrast_threshold", 0.10),
                velocity_scale=vel_scale,
                rng=rng,
            )
        else:
            noisy_dets = simulate_cis(
                gt_sub, fps=info.frame_rate,
                actual_fps=config.get("actual_fps"),
                resolution=sensor.get("resolution"),
                adc_bits=sensor.get("adc_bits", 12),
                velocity_scale=vel_scale,
                rng=rng,
            )

        det_by_frame = {}
        if not noisy_dets.empty:
            for frame_key, frame_grp in noisy_dets.groupby("frame"):
                det_by_frame[int(frame_key)] = frame_grp[["x", "y", "w", "h"]].to_numpy()
        empty_detections = np.empty((0, 4))

        tracker = make_tracker(tracker_kind, use_color=False)
        tracks_per_frame = []
        for frame_num in range(1, max_frames + 1):
            detections = det_by_frame.get(frame_num, empty_detections)
            trks = tracker.update(detections)
            tracks_per_frame.append((frame_num, trks))

        pred_df = tracks_to_df(tracks_per_frame)
        frame_metrics = evaluate(pred_df, gt_sub)
        all_metrics.append(frame_metrics)

    # Average metrics across seeds
    metrics = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if m[key] is not None]
        if vals and isinstance(vals[0], (int, float)):
            metrics[key] = sum(vals) / len(vals)
        else:
            metrics[key] = all_metrics[0][key]

    # Compute power at this velocity scale
    # Mean GT velocity (at scale) for power computation
    gt_vels = gt_sub.copy()
    gt_vels["cx"] = gt_vels["x"] + gt_vels["w"] / 2
    gt_vels = gt_vels.sort_values(["id", "frame"])
    gt_vels["dx"] = gt_vels.groupby("id")["cx"].diff()
    mean_vel = float(gt_vels["dx"].abs().mean() * info.frame_rate * vel_scale)

    if sensor["type"] == "DVS":
        # Event rate ~ perimeter × velocity, use avg object size
        avg_perim = 2 * (gt_sub["w"].mean() + gt_sub["h"].mean())
        event_rate = avg_perim * mean_vel
        eff_rate = min(event_rate, sensor["refractory_cap"])
        actual_power = sensor["p_static_mw"] + eff_rate * sensor["e_per_event_nj"] * 1e-9 * 1e3
        required_fps = None
        actual_fps = None
    else:
        # CIS runs at a configured FPS. Power scales linearly with FPS.
        actual_fps = config.get("actual_fps") or sensor["max_fps"]
        frac = min(actual_fps / sensor["max_fps"], 1.0)
        actual_power = sensor["power_idle_mw"] + frac * (
            sensor["power_max_mw"] - sensor["power_idle_mw"])
        # Required FPS is what you'd NEED for lossless tracking (design-time).
        avg_size = gt_sub["w"].mean()
        required_fps = mean_vel / max(avg_size, 1) * 10

    ct = config.get("contrast_threshold", 0.10)

    # DVS power scales with 1/θ (lower threshold = more events = more power)
    if sensor["type"] == "DVS":
        theta_scale = 0.05 / ct  # baseline 5%
        actual_power = sensor["p_static_mw"] + eff_rate * theta_scale * sensor["e_per_event_nj"] * 1e-9 * 1e3

    return {
        "sensor_name": sensor["name"],
        "sensor_type": sensor["type"],
        "tracker": tracker_kind,
        "velocity_scale": vel_scale,
        "mean_velocity_px_s": round(mean_vel, 1),
        "power_mW": round(actual_power, 2),
        "required_fps": round(required_fps, 1) if required_fps else None,
        "actual_fps": actual_fps,
        "contrast_threshold": ct if sensor["type"] == "DVS" else None,
        "fps_fraction": round((config.get("actual_fps") or sensor.get("max_fps", 0)) / sensor.get("max_fps", 1), 2) if sensor["type"] == "CIS" else None,
        "price": sensor["price"],
        **metrics,
    }


def main():
    import argparse
    sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..", "Ishs-work")))
    from sensor_database import DVS_SENSORS, CIS_SENSORS

    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-root", required=False, default=DEFAULT_MOT_ROOT)
    ap.add_argument("--seq", default="MOT17-04-SDP")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "crossover_results.csv"))
    args = ap.parse_args()

    velocity_scales = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    trackers = ["centroid", "iou", "sort"]

    # Build sensor configs
    sensors = []
    for dvs_sensor in DVS_SENSORS:
        sensors.append({
            "name": dvs_sensor.name, "type": "DVS",
            "pixel_latency_s": dvs_sensor.pixel_latency_us * 1e-6,
            "refractory_cap": dvs_sensor.max_event_rate_mevps * 1e6,
            "resolution": dvs_sensor.resolution,
            "p_static_mw": dvs_sensor.p_static_mw,
            "e_per_event_nj": dvs_sensor.e_per_event_nj,
            "price": dvs_sensor.price_usd,
        })
    for cis_sensor in CIS_SENSORS:
        sensors.append({
            "name": cis_sensor.name, "type": "CIS",
            "max_fps": cis_sensor.max_fps,
            "resolution": cis_sensor.resolution,
            "adc_bits": cis_sensor.adc_bits,
            "power_idle_mw": cis_sensor.power_idle_mw,
            "power_max_mw": cis_sensor.power_at_max_fps_mw,
            "price": cis_sensor.price_usd,
        })

    configs = []
    for sensor in sensors:
        for tracker_kind in trackers:
            for vel_scale in velocity_scales:
                configs.append({
                    "sensor": sensor, "tracker": tracker_kind, "velocity_scale": vel_scale,
                    "mot_root": args.mot_root, "seq": args.seq,
                    "max_frames": args.max_frames, "seed": 42,
                })

    num_configs = len(configs)
    print(f"Running {num_configs} configs ({len(sensors)} sensors × {len(trackers)} trackers "
          f"× {len(velocity_scales)} velocity scales) on {args.workers} workers...\n")
    t_start = time.time()

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, config): config for config in configs}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                rows.append(result)
                if done % 20 == 0 or done == num_configs:
                    print(f"  [{done:3d}/{num_configs}] completed...")
            except Exception as exc:
                failed_config = futures[future]
                print(f"  [{done:3d}/{num_configs}] FAILED {failed_config['sensor']['name']} "
                      f"{failed_config['tracker']} vs={failed_config['velocity_scale']}: {exc}")

    wall = time.time() - t_start
    df = pd.DataFrame(rows)
    df = df.sort_values(["sensor_type", "sensor_name", "tracker", "velocity_scale"])
    df = df.reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"\nDone in {wall:.1f}s. Wrote {len(df)} rows to {args.out}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for col, tk in enumerate(trackers):
        tk_df = df[df["tracker"] == tk]

        ax = axes[0, col]
        for name, grp in tk_df.groupby("sensor_name"):
            grp = grp.sort_values("mean_velocity_px_s")
            style = "--" if grp.iloc[0]["sensor_type"] == "CIS" else "-"
            marker = "s" if grp.iloc[0]["sensor_type"] == "CIS" else "^"
            ax.plot(grp["mean_velocity_px_s"], grp["mota"],
                    marker=marker, linestyle=style, label=name, linewidth=2)
        ax.set_xlabel("Velocity (px/s)")
        ax.set_ylabel("MOTA")
        ax.set_title(f"MOTA — {tk.upper()}")
        ax.set_xscale("log")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        for name, grp in tk_df.groupby("sensor_name"):
            grp = grp.sort_values("mean_velocity_px_s")
            style = "--" if grp.iloc[0]["sensor_type"] == "CIS" else "-"
            marker = "s" if grp.iloc[0]["sensor_type"] == "CIS" else "^"
            ax.plot(grp["mean_velocity_px_s"], grp["power_mW"],
                    marker=marker, linestyle=style, label=name, linewidth=2)
        ax.set_xlabel("Velocity (px/s)")
        ax.set_ylabel("Power (mW)")
        ax.set_title(f"Power — {tk.upper()}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("DVS vs CIS Crossover: All Trackers\n"
                 "(MOT17 trajectories, velocity-scaled, DVS with coasting)",
                 fontweight="bold")
    plt.tight_layout()
    out_png = os.path.join(THIS_DIR, "crossover_plot.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved {out_png}")

    for tk in trackers:
        tk_df = df[df["tracker"] == tk]
        print(f"\n{'=' * 100}")
        print(f"MOTA CROSSOVER — {tk.upper()} tracker")
        print("=" * 100)
        pivot = tk_df.pivot_table(
            index="sensor_name", columns="velocity_scale",
            values="mota", aggfunc="first")
        print(pivot.round(3).to_string())

    print(f"\n{'=' * 100}")
    print("POWER (same for all trackers)")
    print("=" * 100)
    sort_df = df[df["tracker"] == "sort"]
    pivot_p = sort_df.pivot_table(
        index="sensor_name", columns="velocity_scale",
        values="power_mW", aggfunc="first")
    print(pivot_p.round(1).to_string())


if __name__ == "__main__":
    main()
