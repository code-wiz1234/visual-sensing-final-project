"""Run every detector x tracker x sensor combo on MOT17.

Output: full_grid_results.csv
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MOT_ROOT = os.path.join(THIS_DIR, "MOT17")


def _run_one(config: dict) -> dict:
    """Run a single (detector, tracker, sensor) config in a worker process."""
    from ingest_mot import load_seqinfo, load_gt
    from run_benchmark import run_dvs, run_cis

    seq_dir = os.path.join(config["mot_root"], "train", config["seq"])
    info = load_seqinfo(seq_dir)
    gt_df = load_gt(info)

    t0 = time.time()
    sensor = config["sensor"]
    tracker_kind = config["tracker"]
    detector_kind = config["detector"]
    theta = config["theta"]
    max_frames = config["max_frames"]

    if sensor == "DVS":
        r = run_dvs(info, gt_df, theta, max_frames, use_v2e=False,
                    dvs_detector=detector_kind, tracker_kind=tracker_kind)
    else:
        grayscale = sensor == "CIS-gray"
        use_color = (not grayscale) and (tracker_kind == "sort")
        r = run_cis(info, gt_df, max_frames, grayscale=grayscale,
                    use_color_gate=use_color, detector_kind=detector_kind,
                    tracker_kind=tracker_kind)

    return {
        "detector": detector_kind, "tracker": tracker_kind, "sensor": sensor,
        **r, "total_time_s": round(time.time() - t0, 2),
    }


def main():
    import argparse
    BASELINE_THETA = 0.20

    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-root", required=False, default=DEFAULT_MOT_ROOT)
    ap.add_argument("--seq", default="MOT17-04-SDP")
    ap.add_argument("--theta", type=float, default=BASELINE_THETA)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4),
                    help="Max parallel workers")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "full_grid_results.csv"))
    args = ap.parse_args()

    detectors = ["frame_diff", "mog2", "knn", "optical_flow"]
    trackers = ["centroid", "iou", "sort"]
    sensors = ["DVS", "CIS-RGB", "CIS-gray"]

    configs = []
    for detector_kind in detectors:
        for tracker_kind in trackers:
            for sensor in sensors:
                configs.append({
                    "detector": detector_kind, "tracker": tracker_kind, "sensor": sensor,
                    "mot_root": args.mot_root, "seq": args.seq,
                    "theta": args.theta, "max_frames": args.max_frames,
                })

    num_configs = len(configs)
    print(f"Running {num_configs} configs ({len(detectors)} detectors x {len(trackers)} trackers "
          f"x {len(sensors)} sensors) on {args.workers} workers "
          f"({os.cpu_count()} CPU threads available)...")
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
                print(f"  [{done:2d}/{num_configs}] {result['detector']:12s} {result['tracker']:8s} "
                      f"{result['sensor']:8s}  MOTA={result['mota']:+.3f}  "
                      f"IDsw={result['id_switches']:3d}  ({result['total_time_s']:.1f}s)")
            except Exception as exc:
                print(f"  [{done:2d}/{num_configs}] FAILED {failed_config['detector']} {failed_config['tracker']} "
                      f"{failed_config['sensor']}: {exc}")

    wall = time.time() - t_start
    df = pd.DataFrame(rows)

    # Sort: detector → tracker → sensor
    det_order = {det: idx for idx, det in enumerate(detectors)}
    tk_order = {tk: idx for idx, tk in enumerate(trackers)}
    sn_order = {sn: idx for idx, sn in enumerate(sensors)}
    df["_det_rank"] = df["detector"].map(det_order)
    df["_tk_rank"] = df["tracker"].map(tk_order)
    df["_sn_rank"] = df["sensor"].map(sn_order)
    df = df.sort_values(["_det_rank", "_tk_rank", "_sn_rank"]).drop(columns=["_det_rank", "_tk_rank", "_sn_rank"])
    df = df.reset_index(drop=True)
    df.to_csv(args.out, index=False)

    seq_total = sum(row.get("total_time_s", 0) for row in rows)
    print(f"\nDone in {wall:.1f}s wall-clock "
          f"(vs {seq_total:.0f}s sequential — {seq_total/max(wall,1):.1f}x speedup)")
    print(f"Wrote {len(df)} rows to {args.out}\n")

    print("=" * 95)
    print("FULL GRID: detector x tracker x sensor  [same detector+tracker on DVS & CIS]")
    print("=" * 95)
    summary = df[["detector", "tracker", "sensor", "power_mW", "mota",
                  "idf1", "id_switches", "num_pred"]].copy()
    summary["power_mW"] = summary["power_mW"].apply(
        lambda val: f"{val:.1f}" if pd.notna(val) else "—")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
