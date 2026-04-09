# cis_noise_tracking_impact.py
# Harshitha — CIS Noise Impact on Tracking Accuracy
# Team 4: CIS vs DVS Co-Design for Object Tracking
#
# Answers: Does CIS noise hurt tracking performance?
# Connects ModuCIS noise model to Sergey's tracking pipeline.
#
# Method:
#   1. Run Sergey's CIS tracking on CLEAN frames  → MOTA, IDF1, ID switches
#   2. Run Sergey's CIS tracking on NOISY frames  → MOTA, IDF1, ID switches
#   3. Compare: how much does noise degrade tracking?
#   4. Plot results — power vs accuracy for clean vs noisy CIS
#
# Outputs (saved to cis_noise_tracking_results/):
#   cis_nt_accuracy_comparison.png   — MOTA/IDF1 clean vs noisy bar chart
#   cis_nt_power_vs_accuracy.png     — power vs MOTA scatter
#   cis_nt_detections_over_time.png  — detections per frame clean vs noisy
#   cis_nt_summary_table.png         — full metrics table
#   cis_nt_results.csv               — raw results for Sergey

import os, sys, time
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path setup ─────────────────────────────────────────────────────────────
# All Sergey's files copied into this folder — no cross-folder imports needed
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── Import Sergey's pipeline ───────────────────────────────────────────────
try:
    from ingest_mot import load_seqinfo, iter_frames, load_gt
    from cis_detector import MOG2Detector
    from fast_sort import Sort
    from evaluate_tracking import evaluate, tracks_to_df
    print("[OK] Imported Sergey's pipeline")
except ImportError as e:
    raise ImportError(
        f"Missing files in {HERE}\n"
        f"Copy from Sergeys-work: cis_detector.py, fast_sort.py, "
        f"evaluate_tracking.py, fast_eval.py, simple_trackers.py\n"
        f"Error: {e}"
    )

# ── Settings ───────────────────────────────────────────────────────────────
MOT17_SEQ  = os.path.join(HERE, "MOT17", "train", "MOT17-04-SDP")
MAX_FRAMES = 100    # frames to evaluate
CIS_POWER_MW = 332.6   # from ModuCIS worst-case analysis

# ── CIS noise parameters ───────────────────────────────────────────────────
READ_NOISE_E   = 3.0
FULL_WELL_E    = 10000.0
FPN_SIGMA      = 0.01
DARK_CURRENT_E = 5.0
QE             = 0.6

# ── Output directory ───────────────────────────────────────────────────────
OUT_DIR = os.path.join(HERE, 'cis_noise_tracking_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
#  CIS NOISE MODEL
# ══════════════════════════════════════════════════════════════════════════
def apply_cis_noise(frame_bgr, seed=None):
    if seed is not None:
        np.random.seed(seed)
    gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    signal_e = (gray / 255.0) * FULL_WELL_E * QE
    shot_e   = np.random.poisson(np.maximum(signal_e,0)).astype(np.float64) - signal_e
    read_e   = np.random.normal(0, READ_NOISE_E, gray.shape)
    dark_e   = np.random.poisson(DARK_CURRENT_E, gray.shape).astype(np.float64)
    np.random.seed(42)
    fpn_map  = np.random.normal(1.0, FPN_SIGMA, gray.shape)
    np.random.seed(seed if seed is not None else None)
    noisy_e  = (signal_e + shot_e + read_e + dark_e) * fpn_map
    noisy_e  = np.clip(noisy_e, 0, FULL_WELL_E)
    noisy_gray = (noisy_e / FULL_WELL_E * 255.0).astype(np.uint8)
    return cv2.cvtColor(noisy_gray, cv2.COLOR_GRAY2BGR)

# ══════════════════════════════════════════════════════════════════════════
#  RUN TRACKING PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def run_tracking(info, gt_df, apply_noise=False, label="Clean CIS"):
    """
    Run Sergey's MOG2 detector + SORT tracker on frames.
    If apply_noise=True, adds CIS noise before detection.
    Returns metrics dict + per-frame detection counts.
    """
    print(f"\n[CIS Tracking] Running: {label}")
    detector = MOG2Detector(min_area=600, max_area=80000, grayscale=True)
    tracker  = Sort(max_age=5, min_hits=3, iou_threshold=0.3, color_gate=0.0)

    tracks_per_frame  = []
    detections_per_frame = []
    t0 = time.time()

    for frame_idx, frame_bgr in iter_frames(info, max_frames=MAX_FRAMES):
        # Apply noise if requested
        if apply_noise:
            frame_bgr = apply_cis_noise(frame_bgr, seed=frame_idx)

        # Convert to grayscale for detection (same as Sergey's CIS-gray)
        gray        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Detect
        boxes = detector(frame_gray)
        dets  = np.array(boxes) if boxes else np.empty((0, 4))

        # Track
        tracks = tracker.update(dets)
        tracks_per_frame.append((frame_idx, tracks))
        detections_per_frame.append({
            'frame': frame_idx,
            'n_detections': len(boxes),
            'n_tracks': len(tracks),
        })

        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx:3d} | detections={len(boxes):3d} | tracks={len(tracks):3d}")

    wall = time.time() - t0

    # Evaluate against ground truth
    pred_df = tracks_to_df(tracks_per_frame)
    gt_sub  = gt_df[gt_df['frame'] <= MAX_FRAMES]
    metrics = evaluate(pred_df, gt_sub)

    print(f"  Done in {wall:.1f}s | MOTA={metrics['mota']:+.3f} | "
          f"IDF1={metrics['idf1']:.3f} | ID-sw={metrics['id_switches']}")

    return {
        'label':        label,
        'apply_noise':  apply_noise,
        'power_mW':     CIS_POWER_MW,
        'wall_s':       round(wall, 2),
        **metrics,
    }, pd.DataFrame(detections_per_frame)

# ══════════════════════════════════════════════════════════════════════════
#  MAIN — Load data and run both pipelines
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[CIS Tracking] Loading MOT17: {MOT17_SEQ}")
info   = load_seqinfo(MOT17_SEQ)
gt_df  = load_gt(info)
print(f"[CIS Tracking] {info.name} | {info.frame_rate}fps | "
      f"{MAX_FRAMES} frames | {len(gt_df)} GT boxes")

# Run clean CIS
clean_metrics, clean_dets = run_tracking(info, gt_df,
                                          apply_noise=False,
                                          label="Clean CIS")
# Run noisy CIS
noisy_metrics, noisy_dets = run_tracking(info, gt_df,
                                          apply_noise=True,
                                          label="Noisy CIS")

results = [clean_metrics, noisy_metrics]
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUT_DIR, 'cis_nt_results.csv'), index=False)
print(f"\n[Saved] cis_nt_results.csv")

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Accuracy comparison: Clean vs Noisy CIS
# ══════════════════════════════════════════════════════════════════════════
metrics_to_plot = ['mota', 'idf1']
metric_labels   = ['MOTA', 'IDF1']
colors          = ['steelblue', 'tomato']
labels          = ['Clean CIS', 'Noisy CIS']

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('CIS Noise Impact on Tracking Accuracy\n'
             f'MOT17-04-SDP | {MAX_FRAMES} frames | MOG2 + SORT',
             fontsize=13, fontweight='bold')

# MOTA
ax = axes[0]
vals = [clean_metrics['mota'], noisy_metrics['mota']]
bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='black')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:+.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set(title='MOTA\n(Higher = Better)', ylabel='MOTA Score')
ax.set_ylim(min(vals)-0.05, max(vals)+0.08)
ax.axhline(y=0, color='black', lw=0.8, ls='--')
ax.grid(True, alpha=0.3, axis='y')

# IDF1
ax = axes[1]
vals = [clean_metrics['idf1'], noisy_metrics['idf1']]
bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='black')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set(title='IDF1\n(Higher = Better)', ylabel='IDF1 Score')
ax.set_ylim(0, max(vals)+0.08)
ax.grid(True, alpha=0.3, axis='y')

# ID Switches
ax = axes[2]
vals = [clean_metrics['id_switches'], noisy_metrics['id_switches']]
bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='black')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set(title='ID Switches\n(Lower = Better)', ylabel='Count')
ax.set_ylim(0, max(vals)+5)
ax.grid(True, alpha=0.3, axis='y')

# Annotate degradation
delta_mota = noisy_metrics['mota'] - clean_metrics['mota']
delta_idf1 = noisy_metrics['idf1'] - clean_metrics['idf1']
delta_idsw = noisy_metrics['id_switches'] - clean_metrics['id_switches']
fig.text(0.5, 0.02,
         f'Noise effect:  MOTA {delta_mota:+.3f}  |  '
         f'IDF1 {delta_idf1:+.3f}  |  '
         f'ID-switches {delta_idsw:+d}',
         ha='center', fontsize=11, color='darkred', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#fff0f0', edgecolor='red'))

plt.tight_layout(rect=[0,0.08,1,1])
p = os.path.join(OUT_DIR, 'cis_nt_accuracy_comparison.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Power vs Accuracy
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('CIS Power vs Tracking Accuracy\n'
             'Power is constant — noise degrades accuracy without saving power',
             fontsize=12, fontweight='bold')

for m, color, marker in zip(results, colors, ['o', 's']):
    ax.scatter(m['power_mW'], m['mota'],
               s=200, color=color, marker=marker,
               zorder=5, label=m['label'],
               edgecolors='black', linewidth=1.5)
    ax.annotate(f"{m['label']}\nMOTA={m['mota']:+.3f}",
                (m['power_mW'], m['mota']),
                textcoords='offset points', xytext=(10, 5),
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=color))

ax.set(xlabel='CIS Power (mW)', ylabel='MOTA Score')
ax.annotate('← Same power, different accuracy\n   Noise hurts tracking for FREE',
            xy=(CIS_POWER_MW, (clean_metrics['mota']+noisy_metrics['mota'])/2),
            xytext=(CIS_POWER_MW - 40, (clean_metrics['mota']+noisy_metrics['mota'])/2),
            fontsize=9, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred'),
            bbox=dict(boxstyle='round', facecolor='#fff0f0', edgecolor='red'))
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cis_nt_power_vs_accuracy.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Detections per frame: Clean vs Noisy
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('Detections Per Frame — Clean vs Noisy CIS\n'
             'Noise changes how many objects CIS detects each frame',
             fontsize=12, fontweight='bold')

for ax, df_det, color, label in zip(
        axes,
        [clean_dets, noisy_dets],
        colors, labels):
    ax.plot(df_det['frame'], df_det['n_detections'],
            color=color, lw=2, label=f'{label} — detections')
    ax.fill_between(df_det['frame'], df_det['n_detections'],
                    alpha=0.2, color=color)
    ax.axhline(y=df_det['n_detections'].mean(),
               color=color, ls='--', lw=1.5,
               label=f'Mean = {df_det["n_detections"].mean():.1f}')
    ax.set(ylabel='Detections per frame', title=label)
    ax.legend(); ax.grid(True, alpha=0.4)

axes[-1].set_xlabel('Frame Index')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cis_nt_detections_over_time.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 4 — Summary table
# ══════════════════════════════════════════════════════════════════════════
table_data = []
for m in results:
    table_data.append([
        m['label'],
        f"{m['power_mW']:.1f} mW",
        f"{m['mota']:+.4f}",
        f"{m['idf1']:.4f}",
        f"{m['id_switches']}",
        f"{m['num_pred']}",
        f"{m['wall_s']:.1f}s",
    ])
# Delta row
table_data.append([
    'Δ (Noisy − Clean)',
    '0.0 mW',
    f"{delta_mota:+.4f}",
    f"{delta_idf1:+.4f}",
    f"{delta_idsw:+d}",
    f"{noisy_metrics['num_pred']-clean_metrics['num_pred']:+d}",
    '—',
])

col_labels = ['Condition', 'Power', 'MOTA', 'IDF1', 'ID Switches', 'Predictions', 'Time']
row_colors = [['#DDEEFF']*7, ['#FFE8E8']*7, ['#FFF3CD']*7]

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center', loc='center',
    colColours=['#2255AA']*7,
    cellColours=row_colors,
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)
# Header text white
for j in range(len(col_labels)):
    table[0, j].get_text().set_color('white')
    table[0, j].get_text().set_fontweight('bold')
ax.set_title('CIS Noise Impact on Tracking — Summary\n'
             'Same power, same sensor, same pipeline — only noise differs',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cis_nt_summary_table.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('CIS NOISE TRACKING IMPACT SUMMARY')
print('='*65)
print(f'  Sequence   : {info.name}')
print(f'  Frames     : {MAX_FRAMES}')
print(f'  Detector   : MOG2 (same as Sergey)')
print(f'  Tracker    : SORT (same as Sergey)')
print(f'  CIS Power  : {CIS_POWER_MW} mW (constant — noise does NOT change power)')
print(f'\n  {"Metric":<20} {"Clean CIS":>12} {"Noisy CIS":>12} {"Delta":>10}')
print(f'  {"-"*56}')
print(f'  {"MOTA":<20} {clean_metrics["mota"]:>+12.4f} '
      f'{noisy_metrics["mota"]:>+12.4f} {delta_mota:>+10.4f}')
print(f'  {"IDF1":<20} {clean_metrics["idf1"]:>12.4f} '
      f'{noisy_metrics["idf1"]:>12.4f} {delta_idf1:>+10.4f}')
print(f'  {"ID Switches":<20} {clean_metrics["id_switches"]:>12d} '
      f'{noisy_metrics["id_switches"]:>12d} {delta_idsw:>+10d}')
print(f'  {"Power (mW)":<20} {CIS_POWER_MW:>12.1f} {CIS_POWER_MW:>12.1f} {"0.0":>10}')
print(f'\n  KEY FINDING:')
if abs(delta_mota) < 0.01:
    print(f'  CIS noise has MINIMAL impact on tracking ({delta_mota:+.4f} MOTA)')
    print(f'  CIS is robust to sensor noise for pedestrian tracking')
else:
    print(f'  CIS noise DEGRADES tracking by {delta_mota:+.4f} MOTA')
    print(f'  Noise causes {delta_idsw:+d} extra ID switches')
print('\n' + '='*65)
print(f'All outputs in: {OUT_DIR}')
print('  cis_nt_accuracy_comparison.png  ← MOTA/IDF1/ID-sw bar chart')
print('  cis_nt_power_vs_accuracy.png    ← power vs MOTA scatter')
print('  cis_nt_detections_over_time.png ← detections per frame')
print('  cis_nt_summary_table.png        ← full metrics table')
print('  cis_nt_results.csv              ← raw results for Sergey')
