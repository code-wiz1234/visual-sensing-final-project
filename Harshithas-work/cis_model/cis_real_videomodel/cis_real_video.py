# cis_real_video.py
# Harshitha — CIS Noise Model on Real MOT17 Video
# Team 4: CIS vs DVS Co-Design for Object Tracking
#
# Professor feedback: Compare clean CIS vs noisy CIS side by side
# on real video — same as Ishita did for DVS
#
# 4 panel visualization:
#   Panel 1 — Original MOT17 frame (colour)    ← what human sees
#   Panel 2 — Clean CIS (grayscale, no noise)  ← what CIS captures ideally
#   Panel 3 — Noisy CIS (grayscale, with noise)← what CIS physically captures
#   Panel 4 — Difference map (heatmap + gray)  ← where noise deviated from clean
#
# Noise sources modeled:
#   Read noise, Shot noise, FPN, Dark current

import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Path setup ─────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── Import Sergey's MOT17 loader ───────────────────────────────────────────
try:
    from ingest_mot import load_seqinfo, iter_frames
    print("[OK] Imported ingest_mot.py")
except ImportError as e:
    raise ImportError(f"ingest_mot.py not found in {HERE}\nError: {e}")

# ── MOT17 sequence ─────────────────────────────────────────────────────────
MOT17_SEQ  = os.path.join(HERE, "MOT17", "train", "MOT17-04-SDP")
MAX_FRAMES = 50

# ── CIS sensor parameters ──────────────────────────────────────────────────
READ_NOISE_E   = 3.0
FULL_WELL_E    = 10000.0
ADC_BITS       = 10
FPN_SIGMA      = 0.01
DARK_CURRENT_E = 5.0
QE             = 0.6
CIS_POWER_MW   = 332.6   # constant — from ModuCIS worst-case analysis

# ── Output directory ───────────────────────────────────────────────────────
OUT_DIR = os.path.join(HERE, 'cis_real_video_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
#  CIS NOISE MODEL
# ══════════════════════════════════════════════════════════════════════════
def apply_cis_noise(frame_bgr, seed=None):
    if seed is not None:
        np.random.seed(seed)
    gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    signal_e = (gray / 255.0) * FULL_WELL_E * QE

    # Shot noise
    shot_e   = np.random.poisson(np.maximum(signal_e, 0)).astype(np.float64) - signal_e
    # Read noise
    read_e   = np.random.normal(0, READ_NOISE_E, gray.shape)
    # Dark current
    dark_e   = np.random.poisson(DARK_CURRENT_E, gray.shape).astype(np.float64)
    # FPN — fixed pattern (same seed = consistent across frames)
    np.random.seed(42)
    fpn_map  = np.random.normal(1.0, FPN_SIGMA, gray.shape)
    np.random.seed(seed if seed is not None else None)

    noisy_e    = (signal_e + shot_e + read_e + dark_e) * fpn_map
    noisy_e    = np.clip(noisy_e, 0, FULL_WELL_E)
    noisy_gray = (noisy_e / FULL_WELL_E * 255.0).astype(np.uint8)
    noisy_bgr  = cv2.cvtColor(noisy_gray, cv2.COLOR_GRAY2BGR)

    diff         = noisy_gray.astype(np.float64) - gray
    noise_rms    = float(np.sqrt(np.mean(diff**2)))
    signal_mean  = float(np.mean(gray))
    snr_dB       = float(20*np.log10(signal_mean/noise_rms)) if noise_rms > 0 else 60.0

    return noisy_bgr, noisy_gray, gray.astype(np.uint8), diff, {
        'noise_rms':   noise_rms,
        'snr_dB':      snr_dB,
        'shot_rms_e':  float(np.sqrt(np.mean(shot_e**2))),
        'read_rms_e':  float(np.sqrt(np.mean(read_e**2))),
        'fpn_rms_e':   float(np.sqrt(np.mean(((fpn_map-1)*signal_e)**2))),
        'signal_mean': signal_mean,
    }

# ══════════════════════════════════════════════════════════════════════════
#  LOAD FRAMES
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[CIS] Loading MOT17: {MOT17_SEQ}")
info = load_seqinfo(MOT17_SEQ)
print(f"[CIS] {info.name} | {info.im_width}x{info.im_height} | "
      f"{info.frame_rate}fps | {info.seq_length} frames total")

frame_indices = []
original_frames= []   # colour BGR
clean_gray_frames=[]  # grayscale uint8
noisy_bgr_frames= []  # noisy BGR
noisy_gray_frames=[]  # noisy grayscale
diff_frames     = []  # difference (float)
metrics_list    = []

for frame_idx, frame_bgr in iter_frames(info, max_frames=MAX_FRAMES):
    noisy_bgr, noisy_gray, clean_gray, diff, metrics = \
        apply_cis_noise(frame_bgr, seed=frame_idx)
    frame_indices.append(frame_idx)
    original_frames.append(frame_bgr)
    clean_gray_frames.append(clean_gray)
    noisy_bgr_frames.append(noisy_bgr)
    noisy_gray_frames.append(noisy_gray)
    diff_frames.append(diff)
    metrics_list.append(metrics)
    if frame_idx % 10 == 0:
        print(f"  Frame {frame_idx:3d} | SNR={metrics['snr_dB']:.1f}dB | "
              f"Noise RMS={metrics['noise_rms']:.2f}")

print(f"[CIS] Processed {len(frame_indices)} frames.")

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 1 — 4 Panel frame comparison
#  Original (colour) | Clean CIS (gray) | Noisy CIS (gray) | Diff (heatmap + gray)
# ══════════════════════════════════════════════════════════════════════════
sample_indices = [0, len(frame_indices)//2, -1]
sample_labels  = ['Early frame', 'Mid frame', 'Late frame']

for si, label in zip(sample_indices, sample_labels):
    fi  = frame_indices[si]
    m   = metrics_list[si]
    diff= diff_frames[si]

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'CIS Frame Analysis — {label} (Frame {fi})\n'
                 f'Real MOT17-04-SDP  |  SNR={m["snr_dB"]:.1f} dB  |  '
                 f'Noise RMS={m["noise_rms"]:.2f}',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.1)

    # Row 1: 4 panels
    # Panel 1 — Original colour
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(original_frames[si], cv2.COLOR_BGR2RGB))
    ax1.set_title('Original\n(Human View — Colour)', fontsize=9, fontweight='bold')
    ax1.axis('off')

    # Panel 2 — Clean CIS grayscale
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(clean_gray_frames[si], cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Clean CIS\n(Grayscale — No Noise)', fontsize=9, fontweight='bold', color='green')
    ax2.axis('off')

    # Panel 3 — Noisy CIS grayscale
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(noisy_gray_frames[si], cmap='gray', vmin=0, vmax=255)
    ax3.set_title(f'Noisy CIS\n(Grayscale — With Noise)', fontsize=9, fontweight='bold', color='tomato')
    ax3.axis('off')

    # Panel 4 — Difference heatmap (clipped at 95th percentile)
    diff_abs  = np.abs(diff)
    clip_val  = np.percentile(diff_abs, 95)
    diff_clip = np.clip(diff_abs, 0, clip_val)

    ax4 = fig.add_subplot(gs[0, 3])
    im  = ax4.imshow(diff_clip, cmap='jet', vmin=0, vmax=clip_val)
    ax4.set_title('Difference Map\n(Blue=Low Noise, Red=High Noise)', 
                  fontsize=9, fontweight='bold', color='darkred')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label='|Noisy − Clean| (clipped 95th %ile)')

    # Row 2 — grayscale difference
    ax5 = fig.add_subplot(gs[1, :])
    ax5.imshow(diff_clip, cmap='gray', vmin=0, vmax=clip_val)
    ax5.set_title('Difference Map — Grayscale  '
                  '(White = High noise deviation, Black = No deviation)',
                  fontsize=9, fontweight='bold')
    ax5.axis('off')

    p = os.path.join(OUT_DIR, f'cis_rv_4panel_frame{fi}.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Noise RMS over time
# ══════════════════════════════════════════════════════════════════════════
noise_rms_vals = [m['noise_rms'] for m in metrics_list]
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(frame_indices, noise_rms_vals, color='tomato', lw=2, label='Noisy CIS — noise RMS')
ax.axhline(y=0, color='steelblue', lw=2, ls='--', label='Clean CIS — noise RMS = 0')
ax.fill_between(frame_indices, noise_rms_vals, alpha=0.2, color='tomato')
ax.set(title='CIS Noise RMS Over Time — Real MOT17 Video',
       xlabel='Frame Index', ylabel='Noise RMS (pixel intensity)')
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'cis_rv_noise_over_time.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[Saved] cis_rv_noise_over_time.png')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 3 — SNR over time
# ══════════════════════════════════════════════════════════════════════════
snr_vals = [m['snr_dB'] for m in metrics_list]
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(frame_indices, snr_vals, color='tomato', lw=2, label='Noisy CIS SNR')
ax.axhline(y=60, color='steelblue', lw=2, ls='--', label='Clean CIS SNR (~60 dB ideal)')
ax.axhline(y=20, color='red', lw=1.5, ls=':', label='Min acceptable SNR (20 dB)')
ax.set(title='CIS SNR Over Time — Real MOT17 Video',
       xlabel='Frame Index', ylabel='SNR (dB)')
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'cis_rv_snr_over_time.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[Saved] cis_rv_snr_over_time.png')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 4 — CIS power over time (constant)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(y=CIS_POWER_MW, color='steelblue', lw=2.5, ls='--',
           label=f'Clean CIS ({CIS_POWER_MW:.1f} mW)')
ax.axhline(y=CIS_POWER_MW, color='tomato', lw=2.5, ls='-',
           label=f'Noisy CIS ({CIS_POWER_MW:.1f} mW)')
ax.set_xlim(frame_indices[0], frame_indices[-1])
ax.set_ylim(0, CIS_POWER_MW*1.5)
ax.annotate('CIS power is constant regardless\nof noise or scene activity',
            xy=(0.05,0.85), xycoords='axes fraction', fontsize=10, color='darkred',
            bbox=dict(boxstyle='round', facecolor='#fff0f0', edgecolor='red'))
ax.set(title='CIS Power Over Time — Constant regardless of noise',
       xlabel='Frame Index', ylabel='CIS Power (mW)')
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'cis_rv_power_over_time.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[Saved] cis_rv_power_over_time.png')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 5 — Noise breakdown
# ══════════════════════════════════════════════════════════════════════════
shot_vals = [m['shot_rms_e'] for m in metrics_list]
read_vals = [m['read_rms_e'] for m in metrics_list]
fpn_vals  = [m['fpn_rms_e']  for m in metrics_list]
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(frame_indices, shot_vals, color='#2255AA', lw=2, label='Shot noise (e⁻)')
ax.plot(frame_indices, read_vals, color='#EE5522', lw=2, label='Read noise (e⁻)')
ax.plot(frame_indices, fpn_vals,  color='#228B22', lw=2, label='FPN (e⁻)')
ax.set(title='CIS Noise Source Breakdown — Real MOT17 Video',
       xlabel='Frame Index', ylabel='Noise RMS (electrons)')
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'cis_rv_noise_breakdown.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[Saved] cis_rv_noise_breakdown.png')

# ══════════════════════════════════════════════════════════════════════════
#  VIDEO — Original (colour) | Clean CIS (gray) | Noisy CIS (gray) | Diff (heatmap)
# ══════════════════════════════════════════════════════════════════════════
print("\n[CIS] Generating 4-panel comparison video...")
h, w   = clean_gray_frames[0].shape[:2]
label_h= 50
out_w  = w * 4
out_h  = h + label_h
video_path = os.path.join(OUT_DIR, 'cis_rv_comparison_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_path, fourcc, info.frame_rate, (out_w, out_h))

# Pre-compute global diff max for consistent heatmap scale
global_diff_max = max(np.abs(d).max() for d in diff_frames)

for i in range(len(frame_indices)):
    fi   = frame_indices[i]
    m    = metrics_list[i]
    diff = diff_frames[i]

    # Panel 1 — Original colour (resize to same h,w)
    orig  = cv2.resize(original_frames[i], (w, h))

    # Panel 2 — Clean CIS gray → BGR
    clean = cv2.cvtColor(cv2.resize(clean_gray_frames[i], (w,h)), cv2.COLOR_GRAY2BGR)

    # Panel 3 — Noisy CIS gray → BGR
    noisy = cv2.cvtColor(cv2.resize(noisy_gray_frames[i], (w,h)), cv2.COLOR_GRAY2BGR)

    # Panel 4 — Difference heatmap — normalized locally so structure is visible
    diff_abs   = np.abs(diff)
    # Clip at 95th percentile so bright spots don't dominate
    clip_val   = np.percentile(diff_abs, 95)
    diff_clip  = np.clip(diff_abs, 0, clip_val)
    diff_norm  = (diff_clip / clip_val * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(cv2.resize(diff_norm, (w,h)), cv2.COLORMAP_JET)
    # JET: blue=low noise, green=medium, red=high noise

    # Label bar
    bar = np.zeros((label_h, out_w, 3), dtype=np.uint8) + 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, "Original (Human)",   (w//2-100, 32), font, 0.6, (255,255,255), 2)
    cv2.putText(bar, "Clean CIS",          (w+w//2-70, 32), font, 0.6, (150,255,150), 2)
    cv2.putText(bar, f"Noisy CIS SNR={m['snr_dB']:.1f}dB",
                (w*2+w//2-130, 32), font, 0.6, (100,180,255), 2)
    cv2.putText(bar, "Difference (Heatmap)", (w*3+w//2-150, 32), font, 0.6, (100,100,255), 2)
    cv2.putText(bar, f"Frame {fi}", (10, 42), font, 0.45, (180,180,180), 1)

    row = np.concatenate([orig, clean, noisy, diff_color], axis=1)
    writer.write(np.concatenate([bar, row], axis=0))

writer.release()
print(f'[Saved] {video_path}')

# ══════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
avg_snr   = np.mean(snr_vals)
avg_noise = np.mean(noise_rms_vals)
print('\n' + '='*65)
print('CIS REAL VIDEO NOISE MODEL SUMMARY')
print('='*65)
print(f'  Sequence      : {info.name}')
print(f'  Frames used   : {len(frame_indices)} / {info.seq_length}')
print(f'  Resolution    : {info.im_width} x {info.im_height}')
print(f'  Frame rate    : {info.frame_rate} fps')
print(f'\n  Noise parameters:')
print(f'    Read noise  : {READ_NOISE_E} e⁻')
print(f'    FPN sigma   : {FPN_SIGMA*100:.1f}%')
print(f'    Dark current: {DARK_CURRENT_E} e⁻/frame')
print(f'    QE          : {QE*100:.0f}%')
print(f'    ADC bits    : {ADC_BITS}')
print(f'\n  Results (averaged over {len(frame_indices)} frames):')
print(f'    Clean CIS SNR    : ~60 dB (ideal)')
print(f'    Noisy CIS SNR    : {avg_snr:.1f} dB')
print(f'    Avg noise RMS    : {avg_noise:.2f} pixel intensity')
print(f'    CIS power        : {CIS_POWER_MW:.1f} mW (constant)')
print('\n' + '='*65)
print(f'All outputs in: {OUT_DIR}')
print('  cis_rv_4panel_frame[N].png    ← 4 panel comparison (3 frames)')
print('  cis_rv_noise_over_time.png')
print('  cis_rv_snr_over_time.png')
print('  cis_rv_power_over_time.png')
print('  cis_rv_noise_breakdown.png')
print('  cis_rv_comparison_video.mp4   ← Original|Clean|Noisy|Diff video')
