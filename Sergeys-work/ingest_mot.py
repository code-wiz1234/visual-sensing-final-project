"""Load MOT17 sequences (frames, ground truth, detections).

Layout: MOT17/train/<seq>/ with img1/, gt/gt.txt, det/det.txt, seqinfo.ini
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np
import pandas as pd


@dataclass
class SeqInfo:
    name: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    img_dir: str
    gt_path: str
    det_path: str


def load_seqinfo(seq_dir: str) -> SeqInfo:
    ini_path = os.path.join(seq_dir, "seqinfo.ini")
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    seq_cfg = cfg["Sequence"]
    return SeqInfo(
        name=seq_cfg["name"],
        frame_rate=float(seq_cfg["frameRate"]),
        seq_length=int(seq_cfg["seqLength"]),
        im_width=int(seq_cfg["imWidth"]),
        im_height=int(seq_cfg["imHeight"]),
        img_dir=os.path.join(seq_dir, seq_cfg.get("imDir", "img1")),
        gt_path=os.path.join(seq_dir, "gt", "gt.txt"),
        det_path=os.path.join(seq_dir, "det", "det.txt"),
    )


def iter_frames(info: SeqInfo, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, BGR image). frame_idx is 1-based to match MOT17."""
    num_frames = info.seq_length if max_frames is None else min(max_frames, info.seq_length)
    for frame_num in range(1, num_frames + 1):
        path = os.path.join(info.img_dir, f"{frame_num:06d}.jpg")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing frame {path}")
        yield frame_num, img


def load_gt(info: SeqInfo) -> pd.DataFrame:
    """Return ground-truth dataframe with columns: frame, id, x, y, w, h, conf, cls, vis.

    Filtered to pedestrians (cls==1) with conf==1 per MOT17 convention.
    """
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
    df = pd.read_csv(info.gt_path, header=None, names=cols)
    df = df[(df["cls"] == 1) & (df["conf"] == 1)].reset_index(drop=True)
    return df


def load_public_det(info: SeqInfo) -> pd.DataFrame:
    """Load the provided public detections (DPM/FRCNN/SDP depending on subseq).

    Columns: frame, id(-1), x, y, w, h, conf.
    """
    cols = ["frame", "id", "x", "y", "w", "h", "conf"]
    df = pd.read_csv(info.det_path, header=None, usecols=range(7), names=cols)
    return df


def find_sequences(mot_root: str, split: str = "train") -> list[str]:
    """Return sequence directory paths under MOT17/<split>/."""
    split_dir = os.path.join(mot_root, split)
    if not os.path.isdir(split_dir):
        return []
    return sorted(
        os.path.join(split_dir, entry)
        for entry in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, entry))
    )
