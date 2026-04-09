"""Evaluate tracker output against MOT17 ground truth (MOTA, MOTP, IDF1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


def _iou_cost_matrix(gt_boxes, pr_boxes, max_iou=0.5):
    """IoU distance matrix. Returns NaN where IoU is below threshold."""
    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return np.empty((len(gt_boxes), len(pr_boxes)))

    gt_array = np.asarray(gt_boxes, dtype=float)
    pred_array = np.asarray(pr_boxes, dtype=float)

    gt_x2 = gt_array[:, 0] + gt_array[:, 2]
    gt_y2 = gt_array[:, 1] + gt_array[:, 3]
    pred_x2 = pred_array[:, 0] + pred_array[:, 2]
    pred_y2 = pred_array[:, 1] + pred_array[:, 3]

    ix1 = np.maximum(gt_array[:, 0:1], pred_array[:, 0:1].T)
    iy1 = np.maximum(gt_array[:, 1:2], pred_array[:, 1:2].T)
    ix2 = np.minimum(gt_x2[:, None], pred_x2[None, :])
    iy2 = np.minimum(gt_y2[:, None], pred_y2[None, :])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih

    area_g = gt_array[:, 2] * gt_array[:, 3]
    area_p = pred_array[:, 2] * pred_array[:, 3]
    union = area_g[:, None] + area_p[None, :] - inter + 1e-9
    iou = inter / union

    # motmetrics wants distance = 1 - IoU, with NaN for below-threshold
    dist = 1.0 - iou
    dist[iou < max_iou] = np.nan
    return dist


def evaluate(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_threshold: float = 0.5
) -> dict:
    """Return dict with mota, motp, idf1, id_switches, num_gt, num_pred."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)

    # group by frame for fast lookup
    gt_by_frame = dict(list(gt_df.groupby("frame")))
    pred_by_frame = dict(list(pred_df.groupby("frame"))) if not pred_df.empty else {}
    frames = sorted(
        set(gt_df["frame"]).union(pred_df["frame"] if not pred_df.empty else [])
    )

    max_iou = 1.0 - iou_threshold

    for frame_num in frames:
        gt_frame = gt_by_frame.get(frame_num)
        pred_frame = pred_by_frame.get(frame_num)
        gt_ids = gt_frame["id"].tolist() if gt_frame is not None else []
        pr_ids = pred_frame["id"].tolist() if pred_frame is not None else []
        if not gt_ids and not pr_ids:
            continue
        gt_boxes = (
            gt_frame[["x", "y", "w", "h"]].to_numpy() if gt_ids else np.empty((0, 4))
        )
        pr_boxes = (
            pred_frame[["x", "y", "w", "h"]].to_numpy() if pr_ids else np.empty((0, 4))
        )
        dist = _iou_cost_matrix(gt_boxes, pr_boxes, max_iou=max_iou)
        acc.update(gt_ids, pr_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota",
            "motp",
            "idf1",
            "num_switches",
            "num_objects",
            "num_predictions",
        ],
        name="seq",
    )
    row = summary.iloc[0]
    return {
        "mota": float(row["mota"]),
        "motp": float(row["motp"]) if not np.isnan(row["motp"]) else None,
        "idf1": float(row["idf1"]),
        "id_switches": int(row["num_switches"]),
        "num_gt": int(row["num_objects"]),
        "num_pred": int(row["num_predictions"]),
    }


def tracks_to_df(tracks_per_frame):
    """Convert list of (frame_idx, [(tid, x, y, w, h), ...]) to dataframe."""
    rows = []
    for frame_num, tracks in tracks_per_frame:
        for tid, x, y, w, h in tracks:
            rows.append((frame_num, tid, x, y, w, h))
    if not rows:
        return pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h"])
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])
