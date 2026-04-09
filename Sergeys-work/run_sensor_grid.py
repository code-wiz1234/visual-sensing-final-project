"""Run all 8 sensors at native resolution across detectors and trackers.

Output: sensor_grid_results.csv
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MOT_ROOT = os.path.join(THIS_DIR, "MOT17")


def _run_one(config: dict) -> dict:
    """Run one (sensor, detector, tracker) config in a worker process."""
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Ishs-work")))

    import cv2
    import numpy as np
    from ingest_mot import load_seqinfo, iter_frames, load_gt
    from cis_detector import MOG2Detector, KNNDetector, FrameDiffDetector, OpticalFlowDetector
    from event_frames import EventAccumulator, EventDetector
    from v2e_adapter import generate_events
    from evaluate_tracking import evaluate, tracks_to_df
    from run_benchmark import make_tracker, make_detector

    seq_dir = os.path.join(config["mot_root"], "train", config["seq"])
    info = load_seqinfo(seq_dir)
    gt_df = load_gt(info)

    target_w, target_h = config["target_res"]
    orig_w, orig_h = info.im_width, info.im_height
    scale_x, scale_y = target_w / orig_w, target_h / orig_h
    max_frames = config["max_frames"]
    sensor_type = config["sensor_type"]
    detector_kind = config["detector"]
    tracker_kind = config["tracker"]
    theta = config.get("theta", 0.20)

    # Scale ground truth to sensor resolution
    gt_sub = gt_df[gt_df["frame"] <= max_frames].copy()
    gt_sub["x"] = gt_sub["x"] * scale_x
    gt_sub["y"] = gt_sub["y"] * scale_y
    gt_sub["w"] = gt_sub["w"] * scale_x
    gt_sub["h"] = gt_sub["h"] * scale_y

    min_area = max(50, int(200 * scale_x * scale_y))
    max_area = max(2000, int(40000 * scale_x * scale_y))

    t0 = time.time()
    tracks_per_frame = []
    tracker = make_tracker(tracker_kind, use_color=False)

    if sensor_type == "DVS":
        window = max(0.05, 2.0 / info.frame_rate)
        accumulator = EventAccumulator(height=target_h, width=target_w, window_sec=window)

        generic_det = make_detector(detector_kind, grayscale=True,
                                    min_area=min_area, max_area=max_area)
        total_events = 0

        def _resized_frames():
            for fi, bgr in iter_frames(info, max_frames=max_frames):
                small = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                yield fi, small

        frame_idx = 0
        pending_dets = []
        for t_sec, xs, ys, pols in generate_events(
            _resized_frames(), frame_rate=info.frame_rate, theta=theta, use_v2e=False
        ):
            frame_idx += 1
            total_events += len(xs)
            accumulator.add(t_sec, xs, ys, pols)
            if accumulator.ready(t_sec):
                img = accumulator.flush()
                abs_img = np.abs(img).astype(np.float32)
                p = np.percentile(abs_img, 95) if abs_img.max() > 0 else 1.0
                u8 = np.clip(abs_img / max(p, 1e-6) * 200.0, 0, 255).astype(np.uint8)
                bgr_ev = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
                pending_dets = generic_det(bgr_ev)
            dets = np.array(pending_dets) if pending_dets else np.empty((0, 4))
            trks = tracker.update(dets)
            tracks_per_frame.append((frame_idx, trks))

        duration = max(frame_idx / info.frame_rate, 1e-6)
        measured_event_rate = total_events / duration

    else:
        detector = make_detector(detector_kind, grayscale=False,
                                min_area=min_area, max_area=max_area)
        measured_event_rate = None

        for fi, bgr in iter_frames(info, max_frames=max_frames):
            small = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            boxes = detector(small)
            dets = np.array(boxes) if boxes else np.empty((0, 4))
            trks = tracker.update(dets)
            tracks_per_frame.append((fi, trks))

    wall = time.time() - t0
    pred_df = tracks_to_df(tracks_per_frame)
    metrics = evaluate(pred_df, gt_sub)

    return {
        "sensor_name": config["sensor_name"],
        "sensor_type": sensor_type,
        "resolution": f"{target_w}x{target_h}",
        "detector": detector_kind,
        "tracker": tracker_kind,
        "measured_event_rate": measured_event_rate,
        "power_mW": config["power_mw"],
        "price": config["price"],
        "wall_clock_s": round(wall, 2),
        **metrics,
    }


def main():
    import argparse
    sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..", "Ishs-work")))
    from sensor_database import DVS_SENSORS, CIS_SENSORS

    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-root", required=False, default=DEFAULT_MOT_ROOT)
    ap.add_argument("--seq", default="MOT17-04-SDP")
    ap.add_argument("--theta", type=float, default=0.20)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "sensor_grid_results.csv"))
    args = ap.parse_args()

    mot17_event_rate = 762_628
    mot17_fps = 30.0

    detectors = ["frame_diff", "mog2", "knn", "optical_flow"]
    trackers = ["centroid", "iou", "sort"]

    configs = []

    for dvs in DVS_SENSORS:
        for detector_kind in detectors:
            for tracker_kind in trackers:
                configs.append({
                    "sensor_name": dvs.name, "sensor_type": "DVS",
                    "target_res": dvs.resolution, "theta": args.theta,
                    "detector": detector_kind, "tracker": tracker_kind,
                    "power_mw": round(dvs.power_mw(mot17_event_rate), 2),
                    "price": dvs.price_usd,
                    "mot_root": args.mot_root, "seq": args.seq,
                    "max_frames": args.max_frames,
                })

    for cis in CIS_SENSORS:
        for detector_kind in detectors:
            for tracker_kind in trackers:
                configs.append({
                    "sensor_name": cis.name, "sensor_type": "CIS",
                    "target_res": cis.resolution,
                    "detector": detector_kind, "tracker": tracker_kind,
                    "power_mw": round(cis.power_mw(mot17_fps), 2),
                    "price": cis.price_usd,
                    "mot_root": args.mot_root, "seq": args.seq,
                    "max_frames": args.max_frames,
                })

    num_configs = len(configs)
    print(f"Running {num_configs} configs "
          f"({len(DVS_SENSORS)+len(CIS_SENSORS)} sensors × {len(detectors)} detectors "
          f"× {len(trackers)} trackers) on {args.workers} workers "
          f"({os.cpu_count()} threads)...\n")
    t_start = time.time()

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, config): config for config in configs}
        for future in as_completed(futures):
            failed_config = futures[future]
            done += 1
            try:
                result = future.result()
                rows.append(result)
                print(f"  [{done:3d}/{num_configs}] {result['sensor_name']:<22s} "
                      f"{result['detector']:<12s} {result['tracker']:<8s} "
                      f"{result['resolution']:>10s}  MOTA={result['mota']:+.3f}  "
                      f"IDsw={result['id_switches']:3d}  "
                      f"Pwr={result['power_mW']:>7.1f}mW  ({result['wall_clock_s']:.1f}s)")
            except Exception as exc:
                print(f"  [{done:3d}/{num_configs}] FAILED {failed_config['sensor_name']} "
                      f"{failed_config['detector']} {failed_config['tracker']}: {exc}")

    wall = time.time() - t_start
    df = pd.DataFrame(rows)

    # Sort: sensor_type → sensor_name → detector → tracker
    det_order = {det: idx for idx, det in enumerate(detectors)}
    tk_order = {tk: idx for idx, tk in enumerate(trackers)}
    df["_det_rank"] = df["detector"].map(det_order)
    df["_tk_rank"] = df["tracker"].map(tk_order)
    df = df.sort_values(["sensor_type", "sensor_name", "_det_rank", "_tk_rank"])
    df = df.drop(columns=["_det_rank", "_tk_rank"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)

    seq_total = sum(row.get("wall_clock_s", 0) for row in rows)
    print(f"\nDone in {wall:.1f}s wall-clock "
          f"(vs {seq_total:.0f}s sequential — {seq_total/max(wall,1):.1f}x speedup)")
    print(f"Wrote {len(df)} rows to {args.out}\n")

    print("=" * 115)
    print("SENSOR GRID: 8 sensors × 4 detectors × 3 trackers  [same det+tracker on DVS & CIS]")
    print("=" * 115)
    cols = ["sensor_name", "sensor_type", "resolution", "detector", "tracker",
            "power_mW", "mota", "id_switches", "price"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
