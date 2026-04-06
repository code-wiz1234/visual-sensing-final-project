"""End-to-end DVS vs CIS benchmark on a single MOT17 sequence.

Runs three pipelines (DVS, CIS-RGB, CIS-gray), evaluates MOTA/MOTP/IDF1,
and computes sensor power from measured event rates and frame rates.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

from sensor_database import DVS_SENSORS, CIS_SENSORS
from ingest_mot import load_seqinfo, iter_frames, load_gt, load_public_det
from v2e_adapter import generate_events, is_v2e_available
from event_frames import EventAccumulator, EventDetector
from cis_detector import (
    MOG2Detector,
    KNNDetector,
    FrameDiffDetector,
    OpticalFlowDetector,
    PublicDetections,
)
from fast_sort import Sort
from sort_tracker import color_histogram
from simple_trackers import CentroidTracker, IoUTracker
from evaluate_tracking import evaluate, tracks_to_df

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MOT_ROOT = os.path.join(THIS_DIR, "MOT17")
BASELINE_THETA = 0.20  # default contrast threshold (log-intensity units)


def make_detector(
    kind: str, grayscale: bool = False, min_area: int = 600, max_area: int = 80000
):
    """Factory for detectors that work on both event images and intensity frames."""
    if kind == "frame_diff":
        return FrameDiffDetector(
            threshold=30, min_area=min_area, max_area=max_area, grayscale=grayscale
        )
    if kind == "mog2":
        return MOG2Detector(min_area=min_area, max_area=max_area, grayscale=grayscale)
    if kind == "knn":
        return KNNDetector(min_area=min_area, max_area=max_area, grayscale=grayscale)
    if kind == "optical_flow":
        return OpticalFlowDetector(
            mag_threshold=2.0, min_area=min_area, max_area=max_area, grayscale=grayscale
        )
    raise ValueError(f"unknown detector: {kind}")


def make_tracker(kind: str, use_color: bool = False):
    """Factory: all three trackers expose update(dets, histograms=None)."""
    if kind == "centroid":
        return CentroidTracker(max_distance=80.0, max_age=5, min_hits=3)
    if kind == "iou":
        return IoUTracker(iou_threshold=0.3, max_age=5, min_hits=3)
    if kind == "sort":
        color_gate = 0.5 if use_color else 0.0
        return Sort(max_age=5, min_hits=3, iou_threshold=0.3, color_gate=color_gate)
    raise ValueError(f"unknown tracker: {kind}")


def run_dvs(
    info,
    gt_df,
    theta: float,
    max_frames: int,
    use_v2e: bool,
    dvs_detector: str = "cc",
    tracker_kind: str = "sort",
    sensor=None,
) -> dict:
    """Run DVS pipeline on a sequence.

    dvs_detector: "cc" = connected components on event image (DVS-native),
                  "mog2" = apply MOG2 to the event image (same algorithm as
                           the CIS side, making detector comparisons fair).
    """
    t0 = time.time()
    # Accumulate ~2 frames worth of events. MOT17 is 30 fps (33 ms/frame), so a
    # 50 ms window catches motion across frame pairs without smearing distinct
    # walkers together.
    window = max(0.05, 2.0 / info.frame_rate)
    accumulator = EventAccumulator(
        height=info.im_height, width=info.im_width, window_sec=window
    )
    if dvs_detector == "cc":
        # DVS-native connected-components detector on raw event image
        detector = EventDetector(min_area=600, max_area=80000, dilate_iters=3)
        generic_det = None
    else:
        # Use same detector class as CIS side on the event-accumulated image
        # (rescaled to uint8) so both sides use the same detector.
        detector = None
        generic_det = make_detector(dvs_detector, grayscale=True)
    tracker = make_tracker(tracker_kind, use_color=False)

    total_events = 0
    total_time = 0.0
    tracks_per_frame = []
    pending_dets: list[tuple[float, float, float, float]] = []

    frame_idx = 0
    for t_sec, xs, ys, pols in generate_events(
        iter_frames(info, max_frames=max_frames),
        frame_rate=info.frame_rate,
        theta=theta,
        use_v2e=use_v2e,
    ):
        frame_idx += 1
        total_events += len(xs)
        total_time = t_sec if t_sec > total_time else total_time
        accumulator.add(t_sec, xs, ys, pols)
        if accumulator.ready(t_sec):
            img = accumulator.flush()
            if generic_det is not None:
                # Event image -> uint8 [0,255] intensity, then feed the same
                # detector class the CIS side uses. Apples-to-apples.
                abs_img = np.abs(img).astype(np.float32)
                p = np.percentile(abs_img, 95) if abs_img.max() > 0 else 1.0
                u8 = np.clip(abs_img / max(p, 1e-6) * 200.0, 0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
                pending_dets = generic_det(bgr)
            else:
                pending_dets = detector(img)
        dets = np.array(pending_dets) if pending_dets else np.empty((0, 4))
        tracks = tracker.update(dets)
        tracks_per_frame.append((frame_idx, tracks))

    duration = max(total_time, max_frames / info.frame_rate, 1e-6)
    measured_rate = total_events / duration  # events / sec

    dvs_sensor = sensor or DVS_SENSORS[0]  # default: Lichtsteiner 2008
    power_mW = dvs_sensor.power_mw(measured_rate)

    pred_df = tracks_to_df(tracks_per_frame)
    gt_sub = gt_df[gt_df["frame"] <= max_frames] if max_frames else gt_df
    metrics = evaluate(pred_df, gt_sub)

    return {
        "sensor": dvs_sensor.name,
        "theta": theta,
        "measured_event_rate": measured_rate,
        "measured_fps": info.frame_rate,
        "power_mW": round(power_mW, 2),
        "wall_clock_s": round(time.time() - t0, 2),
        **metrics,
    }


def run_cis(
    info,
    gt_df,
    max_frames: int,
    grayscale: bool,
    use_color_gate: bool,
    detector_kind: str,
    tracker_kind: str = "sort",
    sensor=None,
) -> dict:
    t0 = time.time()

    if detector_kind == "public":
        det_df = load_public_det(info)
        det_fn = PublicDetections(det_df)
        get_boxes = lambda fi, fr: det_fn(fi, fr)  # noqa: E731
    else:
        det_obj = make_detector(detector_kind, grayscale=grayscale)
        get_boxes = lambda fi, fr: det_obj(fr)  # noqa: E731

    use_color = use_color_gate and not grayscale
    tracker = make_tracker(tracker_kind, use_color=use_color)

    tracks_per_frame = []
    for frame_idx, bgr in iter_frames(info, max_frames=max_frames):
        frame_for_det = bgr
        if grayscale:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            frame_for_det = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        boxes = get_boxes(frame_idx, frame_for_det)
        dets = np.array(boxes) if boxes else np.empty((0, 4))
        hists = None
        if use_color and tracker_kind == "sort" and len(boxes):
            hists = [color_histogram(bgr, box) for box in boxes]
        tracks = tracker.update(dets, histograms=hists)
        tracks_per_frame.append((frame_idx, tracks))

    pred_df = tracks_to_df(tracks_per_frame)
    gt_sub = gt_df[gt_df["frame"] <= max_frames] if max_frames else gt_df
    metrics = evaluate(pred_df, gt_sub)

    cis_sensor = sensor or CIS_SENSORS[1]  # default: IMX327 1080p
    power_mW = cis_sensor.power_mw(info.frame_rate)
    label = f"CIS-{'gray' if grayscale else 'RGB'} ({cis_sensor.name})"

    return {
        "sensor": label,
        "theta": None,
        "measured_event_rate": None,
        "measured_fps": info.frame_rate,
        "power_mW": round(power_mW, 2),
        "wall_clock_s": round(time.time() - t0, 2),
        **metrics,
    }


def run_sequence(
    mot_root: str,
    seq: str,
    theta: float,
    max_frames: int,
    detector_kind: str,
    use_v2e: bool,
    dvs_detector: str = "cc",
    tracker_kind: str = "sort",
) -> pd.DataFrame:
    seq_dir = os.path.join(mot_root, "train", seq)
    info = load_seqinfo(seq_dir)
    gt_df = load_gt(info)

    rows = []
    tag = f"v2e={use_v2e and is_v2e_available()}, dvs_det={dvs_detector}, tracker={tracker_kind}"
    print(f"[{seq}] DVS @ theta={theta} ({tag})")
    rows.append(
        {
            "sequence": seq,
            "tracker": tracker_kind,
            **run_dvs(
                info, gt_df, theta, max_frames, use_v2e, dvs_detector, tracker_kind
            ),
        }
    )
    print(f"[{seq}] CIS-RGB ({detector_kind}, tracker={tracker_kind})")
    rows.append(
        {
            "sequence": seq,
            "tracker": tracker_kind,
            **run_cis(
                info,
                gt_df,
                max_frames,
                grayscale=False,
                use_color_gate=True,
                detector_kind=detector_kind,
                tracker_kind=tracker_kind,
            ),
        }
    )
    print(f"[{seq}] CIS-gray ({detector_kind}, tracker={tracker_kind})")
    rows.append(
        {
            "sequence": seq,
            "tracker": tracker_kind,
            **run_cis(
                info,
                gt_df,
                max_frames,
                grayscale=True,
                use_color_gate=False,
                detector_kind=detector_kind,
                tracker_kind=tracker_kind,
            ),
        }
    )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mot-root",
        required=False,
        default=DEFAULT_MOT_ROOT,
        help="Path to MOT17 root (containing train/)",
    )
    ap.add_argument("--seq", default="MOT17-04-SDP", help="Sequence name, e.g. MOT17-04-SDP")
    ap.add_argument("--theta", type=float, default=BASELINE_THETA)
    ap.add_argument(
        "--thetas",
        nargs="*",
        type=float,
        default=None,
        help="Sweep multiple thetas (overrides --theta)",
    )
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument(
        "--detector",
        choices=["mog2", "public"],
        default="mog2",
        help="CIS-side detector",
    )
    ap.add_argument(
        "--dvs-detector",
        choices=["cc", "mog2"],
        default="cc",
        help="DVS-side detector: cc = connected components on event "
        "image (DVS-native); mog2 = MOG2 on event image, "
        "same algorithm as --detector mog2 for fair comparison",
    )
    ap.add_argument(
        "--tracker",
        choices=["centroid", "iou", "sort"],
        default="sort",
        help="Tracking algorithm (applied to both DVS and CIS)",
    )
    ap.add_argument(
        "--trackers",
        nargs="*",
        default=None,
        help="Sweep multiple trackers (overrides --tracker)",
    )
    ap.add_argument(
        "--no-v2e",
        action="store_true",
        help="Force the frame-diff fallback instead of v2e",
    )
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "real_world_results.csv"))
    args = ap.parse_args()

    thetas = args.thetas if args.thetas else [args.theta]
    trackers = args.trackers if args.trackers else [args.tracker]
    seqs = args.seq.split(",")
    all_rows = []
    for seq in seqs:
        for th in thetas:
            for tk in trackers:
                df = run_sequence(
                    args.mot_root,
                    seq.strip(),
                    th,
                    args.max_frames,
                    args.detector,
                    use_v2e=not args.no_v2e,
                    dvs_detector=args.dvs_detector,
                    tracker_kind=tk,
                )
                all_rows.append(df)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
