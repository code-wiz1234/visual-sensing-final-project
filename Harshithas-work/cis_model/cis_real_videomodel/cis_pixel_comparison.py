# cis_pixel_comparison.py
# Harshitha — CIS Pixel-by-Pixel Comparison on Real MOT17 Video
# Team 4: CIS vs DVS Co-Design for Object Tracking
#
# Professor feedback: Pixel by pixel comparison of:
#   - Original MOT17 frame (what human sees)
#   - Clean CIS capture (grayscale, no noise)
#   - Noisy CIS capture (grayscale, with noise)
#
# Outputs:
#   cis_pixel_results/
#     cis_px_scatter.png         — pixel values: original vs clean vs noisy scatter
#     cis_px_histogram.png       — distribution of pixel values all three
#     cis_px_error_histogram.png — distribution of noise error (noisy - clean)
#     cis_px_zoom_grid.png       — zoomed pixel grid showing exact values
#     cis_px_stats_table.png     — summary stats table per region

import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

# ── Path setup ─────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    from ingest_mot import load_seqinfo, iter_frames
    print("[OK] Imported ingest_mot.py")
except ImportError as e:
    raise ImportError(f"ingest_mot.py not found in {HERE}\nError: {e}")

# ── CIS noise parameters (same as cis_real_video.py) ──────────────────────
READ_NOISE_E   = 3.0
FULL_WELL_E    = 10000.0
FPN_SIGMA      = 0.01
DARK_CURRENT_E = 5.0
QE             = 0.6

# ── Settings ───────────────────────────────────────────────────────────────
MOT17_SEQ    = os.path.join(HERE, "MOT17", "train", "MOT17-04-SDP")
FRAME_TO_USE = 25      # which frame to analyse pixel by pixel
ZOOM_X       = 200     # top-left x of zoom region
ZOOM_Y       = 150     # top-left y of zoom region
ZOOM_W       = 20      # zoom region width  (pixels)
ZOOM_H       = 20      # zoom region height (pixels)
SAMPLE_STEP  = 8       # sample every Nth pixel for scatter (keeps plot readable)

OUT_DIR = os.path.join(HERE, 'cis_pixel_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
#  CIS NOISE MODEL (same as cis_real_video.py)
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
    clean_gray = gray.astype(np.uint8)
    return clean_gray, noisy_gray

# ══════════════════════════════════════════════════════════════════════════
#  LOAD ONE FRAME
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[CIS Pixel] Loading frame {FRAME_TO_USE} from {MOT17_SEQ}")
info = load_seqinfo(MOT17_SEQ)

original_bgr = None
for frame_idx, frame_bgr in iter_frames(info, max_frames=FRAME_TO_USE):
    if frame_idx == FRAME_TO_USE:
        original_bgr = frame_bgr
        break

if original_bgr is None:
    raise RuntimeError(f"Frame {FRAME_TO_USE} not found")

# Convert original to grayscale for fair comparison
original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
clean_gray, noisy_gray = apply_cis_noise(original_bgr, seed=FRAME_TO_USE)

print(f"[CIS Pixel] Frame shape: {original_gray.shape}")
print(f"[CIS Pixel] Original mean: {original_gray.mean():.1f}")
print(f"[CIS Pixel] Clean CIS mean: {clean_gray.mean():.1f}")
print(f"[CIS Pixel] Noisy CIS mean: {noisy_gray.mean():.1f}")

# Flatten for pixel-level analysis
orig_flat  = original_gray.flatten().astype(np.float64)
clean_flat = clean_gray.flatten().astype(np.float64)
noisy_flat = noisy_gray.flatten().astype(np.float64)
error_flat = noisy_flat - clean_flat   # noise error per pixel

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Scatter: Original vs Clean vs Noisy pixel values
# ══════════════════════════════════════════════════════════════════════════
# Sample every Nth pixel to keep scatter readable
idx = np.arange(0, len(orig_flat), SAMPLE_STEP)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'CIS Pixel-by-Pixel Comparison — Frame {FRAME_TO_USE}\n'
             f'MOT17-04-SDP  |  Every {SAMPLE_STEP}th pixel sampled',
             fontsize=12, fontweight='bold')

# Left — Original vs Clean CIS
ax = axes[0]
ax.scatter(orig_flat[idx], clean_flat[idx],
           s=0.5, alpha=0.3, color='steelblue', label='Clean CIS vs Original')
ax.plot([0,255],[0,255], 'r--', lw=1.5, label='Perfect match (y=x)')
ax.set(title='Original vs Clean CIS\n(Should be near perfect)',
       xlabel='Original pixel value', ylabel='Clean CIS pixel value')
ax.legend(markerscale=10); ax.grid(True, alpha=0.3)

# Right — Original vs Noisy CIS
ax = axes[1]
sc = ax.scatter(orig_flat[idx], noisy_flat[idx],
                s=0.5, alpha=0.3, color='tomato', label='Noisy CIS vs Original')
ax.plot([0,255],[0,255], 'b--', lw=1.5, label='Perfect match (y=x)')
ax.set(title='Original vs Noisy CIS\n(Scatter shows noise deviation)',
       xlabel='Original pixel value', ylabel='Noisy CIS pixel value')
ax.legend(markerscale=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
p = os.path.join(OUT_DIR, 'cis_px_scatter.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Histogram: distribution of pixel values
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
bins = np.arange(0, 256, 2)
ax.hist(orig_flat,  bins=bins, alpha=0.5, color='#2255AA',
        label='Original (human view)', density=True)
ax.hist(clean_flat, bins=bins, alpha=0.5, color='steelblue',
        label='Clean CIS', density=True, linestyle='--',
        histtype='step', linewidth=2)
ax.hist(noisy_flat, bins=bins, alpha=0.5, color='tomato',
        label='Noisy CIS', density=True, linestyle=':',
        histtype='step', linewidth=2)
ax.set(title=f'Pixel Value Distribution — Frame {FRAME_TO_USE}\n'
             'Clean CIS matches original — Noisy CIS distribution spreads wider',
       xlabel='Pixel Intensity (0=black, 255=white)',
       ylabel='Density')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cis_px_histogram.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Error histogram: distribution of noise error (noisy - clean)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(error_flat, bins=100, color='tomato', alpha=0.7,
        label='Noise error (Noisy − Clean)', density=True)
ax.axvline(x=0,  color='black',     lw=2,   label='Zero error')
ax.axvline(x=error_flat.mean(),  color='blue', lw=2, ls='--',
           label=f'Mean error = {error_flat.mean():.2f}')
ax.axvline(x=error_flat.std(),   color='red',  lw=1.5, ls=':',
           label=f'Std = {error_flat.std():.2f}')
ax.axvline(x=-error_flat.std(),  color='red',  lw=1.5, ls=':')
ax.set(title=f'CIS Noise Error Distribution — Frame {FRAME_TO_USE}\n'
             'Centred near zero = unbiased noise  |  Width = noise magnitude',
       xlabel='Noise Error (Noisy pixel − Clean pixel)',
       ylabel='Density')
ax.legend(); ax.grid(True, alpha=0.3)
# Annotate stats
ax.text(0.72, 0.85,
        f'Mean  = {error_flat.mean():.3f}\n'
        f'Std   = {error_flat.std():.3f}\n'
        f'RMS   = {np.sqrt(np.mean(error_flat**2)):.3f}\n'
        f'Max   = {error_flat.max():.1f}\n'
        f'Min   = {error_flat.min():.1f}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#fff0f0', edgecolor='tomato'))
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cis_px_error_histogram.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 4 — Zoomed pixel grid: exact pixel values side by side
#  Shows a small region (ZOOM_W x ZOOM_H) with actual numbers
# ══════════════════════════════════════════════════════════════════════════
orig_zoom  = original_gray[ZOOM_Y:ZOOM_Y+ZOOM_H, ZOOM_X:ZOOM_X+ZOOM_W]
clean_zoom = clean_gray[ZOOM_Y:ZOOM_Y+ZOOM_H,    ZOOM_X:ZOOM_X+ZOOM_W]
noisy_zoom = noisy_gray[ZOOM_Y:ZOOM_Y+ZOOM_H,    ZOOM_X:ZOOM_X+ZOOM_W]
diff_zoom  = noisy_zoom.astype(np.float64) - clean_zoom.astype(np.float64)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(f'CIS Zoomed Pixel Grid — Region [{ZOOM_X}:{ZOOM_X+ZOOM_W}, '
             f'{ZOOM_Y}:{ZOOM_Y+ZOOM_H}]  Frame {FRAME_TO_USE}\n'
             f'Each cell shows exact pixel value',
             fontsize=12, fontweight='bold')

def draw_pixel_grid(ax, data, title, cmap, vmin, vmax, fmt='{:.0f}'):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='auto')
    # Annotate each cell with its value
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            val = data[y, x]
            text_color = 'white' if val < (vmax+vmin)/2 else 'black'
            ax.text(x, y, fmt.format(val),
                    ha='center', va='center',
                    fontsize=5.5, color=text_color, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Pixel X'); ax.set_ylabel('Pixel Y')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

draw_pixel_grid(axes[0,0], orig_zoom,
                'Original (Human View)\nActual pixel intensities',
                'gray', 0, 255)
draw_pixel_grid(axes[0,1], clean_zoom,
                'Clean CIS\nGrayscale — no noise',
                'gray', 0, 255)
draw_pixel_grid(axes[1,0], noisy_zoom,
                'Noisy CIS\nGrayscale — with noise',
                'gray', 0, 255)

clip = np.percentile(np.abs(diff_zoom), 95) or 1
draw_pixel_grid(axes[1,1], diff_zoom,
                'Difference (Noisy − Clean)\nRed=positive, Blue=negative',
                'RdBu_r', -clip, clip, fmt='{:+.0f}')

plt.tight_layout()
p = os.path.join(OUT_DIR, 'cis_px_zoom_grid.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT 5 — Stats table per image region
#  Divides frame into 3x3 grid, shows stats per region
# ══════════════════════════════════════════════════════════════════════════
h, w   = original_gray.shape
rows_g = 3; cols_g = 3
rh = h // rows_g; rw = w // cols_g

region_stats = []
region_labels = []
for r in range(rows_g):
    for c in range(cols_g):
        y0,y1 = r*rh, (r+1)*rh
        x0,x1 = c*rw, (c+1)*rw
        orig_r  = original_gray[y0:y1, x0:x1].astype(np.float64)
        noisy_r = noisy_gray[y0:y1, x0:x1].astype(np.float64)
        clean_r = clean_gray[y0:y1, x0:x1].astype(np.float64)
        err_r   = noisy_r - clean_r
        region_stats.append([
            f'R{r+1}C{c+1}',
            f'{orig_r.mean():.1f}',
            f'{clean_r.mean():.1f}',
            f'{noisy_r.mean():.1f}',
            f'{np.sqrt(np.mean(err_r**2)):.2f}',
            f'{20*np.log10(orig_r.mean()/max(np.sqrt(np.mean(err_r**2)),0.01)):.1f}',
        ])
        region_labels.append(f'R{r+1}C{c+1}')

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')
col_labels = ['Region', 'Orig Mean', 'Clean Mean', 'Noisy Mean',
              'Noise RMS', 'SNR (dB)']
colors_hdr = ['#2255AA'] * len(col_labels)
table = ax.table(
    cellText=region_stats,
    colLabels=col_labels,
    cellLoc='center', loc='center',
    colColours=['#DDEEFF']*len(col_labels)
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
ax.set_title(f'CIS Per-Region Pixel Statistics — Frame {FRAME_TO_USE}\n'
             f'Frame divided into {rows_g}x{cols_g} grid',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cis_px_stats_table.png')
plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
print(f'[Saved] {p}')

# ══════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('CIS PIXEL COMPARISON SUMMARY')
print('='*65)
print(f'  Frame analysed  : {FRAME_TO_USE}')
print(f'  Total pixels    : {len(orig_flat):,}')
print(f'  Zoom region     : [{ZOOM_X}:{ZOOM_X+ZOOM_W}, {ZOOM_Y}:{ZOOM_Y+ZOOM_H}]')
print(f'\n  Pixel statistics (full frame):')
print(f'    Original mean          : {orig_flat.mean():.2f}')
print(f'    Clean CIS mean         : {clean_flat.mean():.2f}')
print(f'    Noisy CIS mean         : {noisy_flat.mean():.2f}')
print(f'    Noise error mean       : {error_flat.mean():.4f}  (near 0 = unbiased)')
print(f'    Noise error std        : {error_flat.std():.4f}')
print(f'    Noise RMS              : {np.sqrt(np.mean(error_flat**2)):.4f}')
print(f'    % pixels with |err|>5  : {(np.abs(error_flat)>5).mean()*100:.1f}%')
print(f'    % pixels with |err|>10 : {(np.abs(error_flat)>10).mean()*100:.1f}%')
print('\n' + '='*65)
print(f'All outputs in: {OUT_DIR}')
print('  cis_px_scatter.png       ← original vs clean vs noisy scatter')
print('  cis_px_histogram.png     ← pixel value distributions')
print('  cis_px_error_histogram.png ← noise error distribution')
print('  cis_px_zoom_grid.png     ← exact pixel values in zoomed region')
print('  cis_px_stats_table.png   ← per-region stats table')
