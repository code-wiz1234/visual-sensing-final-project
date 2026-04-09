"""Fast MOTA/MOTP/IDF1 calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _iou_matrix(gt_xywh: np.ndarray, pr_xywh: np.ndarray) -> np.ndarray:
    """IoU between two sets of xywh bounding boxes."""
    if len(gt_xywh) == 0 or len(pr_xywh) == 0:
        return np.empty((len(gt_xywh), len(pr_xywh)))

    gt = gt_xywh
    pred = pr_xywh
    gt_x2 = gt[:, 0] + gt[:, 2]
    gt_y2 = gt[:, 1] + gt[:, 3]
    pred_x2 = pred[:, 0] + pred[:, 2]
    pred_y2 = pred[:, 1] + pred[:, 3]

    ix1 = np.maximum(gt[:, 0:1], pred[:, 0:1].T)
    iy1 = np.maximum(gt[:, 1:2], pred[:, 1:2].T)
    ix2 = np.minimum(gt_x2[:, None], pred_x2[None, :])
    iy2 = np.minimum(gt_y2[:, None], pred_y2[None, :])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih
    area_g = gt[:, 2] * gt[:, 3]
    area_p = pred[:, 2] * pred[:, 3]
    union = area_g[:, None] + area_p[None, :] - inter + 1e-9
    return inter / union


def evaluate(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
             iou_threshold: float = 0.5) -> dict:
    """Compute MOTA, MOTP, IDF1, id_switches, num_gt, num_pred."""
    from scipy.optimize import linear_sum_assignment

    # Pre-group by frame
    gt_by_frame: dict[int, tuple[list, np.ndarray]] = {}
    for frame_num, grp in gt_df.groupby("frame"):
        gt_by_frame[int(frame_num)] = (grp["id"].tolist(), grp[["x", "y", "w", "h"]].to_numpy())

    pr_by_frame: dict[int, tuple[list, np.ndarray]] = {}
    if not pred_df.empty:
        for frame_num, grp in pred_df.groupby("frame"):
            pr_by_frame[int(frame_num)] = (grp["id"].tolist(), grp[["x", "y", "w", "h"]].to_numpy())

    frames = sorted(set(gt_by_frame.keys()) | set(pr_by_frame.keys()))

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    total_dist = 0.0  # sum of (1-IoU) for matched pairs (for MOTP)
    total_matches = 0

    # For IDF1: track how many GT frames each GT-id and pred-id appears,
    # and how many frames they are correctly matched
    gt_id_frames: dict[int, int] = {}
    pr_id_frames: dict[int, int] = {}
    match_count: dict[tuple[int, int], int] = {}

    # Track last-frame assignment for ID-switch detection
    prev_match: dict[int, int] = {}  # gt_id -> pred_id from previous frame

    for frame_num in frames:
        gt_ids, gt_boxes = gt_by_frame.get(frame_num, ([], np.empty((0, 4))))
        pr_ids, pr_boxes = pr_by_frame.get(frame_num, ([], np.empty((0, 4))))
        n_gt = len(gt_ids)
        n_pr = len(pr_ids)
        total_gt += n_gt
        total_pred += n_pr

        # Count frames per ID (for IDF1)
        for gid in gt_ids:
            gt_id_frames[gid] = gt_id_frames.get(gid, 0) + 1
        for pid in pr_ids:
            pr_id_frames[pid] = pr_id_frames.get(pid, 0) + 1

        if n_gt == 0:
            total_fp += n_pr
            continue
        if n_pr == 0:
            total_fn += n_gt
            continue

        # Compute IoU matrix and match
        iou = _iou_matrix(gt_boxes, pr_boxes)
        cost = 1.0 - iou
        cost[iou < iou_threshold] = 1e6  # forbid low-IoU matches

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pr = set()
        for gt_idx, pred_idx in zip(row_ind, col_ind):
            if iou[gt_idx, pred_idx] >= iou_threshold:
                matched_gt.add(gt_idx)
                matched_pr.add(pred_idx)
                total_tp += 1
                total_dist += (1.0 - iou[gt_idx, pred_idx])
                total_matches += 1

                gid = gt_ids[gt_idx]
                pid = pr_ids[pred_idx]
                key = (gid, pid)
                match_count[key] = match_count.get(key, 0) + 1

                # ID switch: same GT was matched to a different pred last frame
                if gid in prev_match and prev_match[gid] != pid:
                    total_idsw += 1
                prev_match[gid] = pid

        total_fn += n_gt - len(matched_gt)
        total_fp += n_pr - len(matched_pr)

    # MOTA = 1 - (FN + FP + IDSW) / total_gt
    mota = 1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)

    # MOTP = avg distance for matched pairs
    motp = total_dist / max(total_matches, 1) if total_matches > 0 else None

    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    # For each (gt_id, pr_id) pair, find the best match and count
    # Simplified: for each gt_id, its best-matched pr_id gets IDTP = match_count
    idtp = 0
    gt_matched = set()
    pr_matched = set()
    # Sort by match count descending for greedy best-match assignment
    sorted_pairs = sorted(match_count.items(), key=lambda x: -x[1])
    for (gid, pid), cnt in sorted_pairs:
        if gid not in gt_matched and pid not in pr_matched:
            idtp += cnt
            gt_matched.add(gid)
            pr_matched.add(pid)

    total_gt_frames = sum(gt_id_frames.values())
    total_pr_frames = sum(pr_id_frames.values())
    idfn = total_gt_frames - idtp
    idfp = total_pr_frames - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    return {
        "mota": float(mota),
        "motp": float(motp) if motp is not None else None,
        "idf1": float(idf1),
        "id_switches": int(total_idsw),
        "num_gt": int(total_gt),
        "num_pred": int(total_pred),
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
