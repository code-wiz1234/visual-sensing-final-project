# run_all_scenes_cis.py
# One-click runner for ModuCIS (Conventional CIS) across ALL scene-model pairs
# Usage (from repo root or from Use_cases/):
#   python3 run_all_scenes_cis.py
#
# What it does
#  - Imports Ramaa's scene model (visualcomputing.py): object_sizes, velocities, safety_factor
#  - Uses analytical rule: fps_min = (velocity / object_size) * safety_factor
#  - (NEW) Computes ADC bits from scene: contrast -> SNR_dB -> ENOB -> bits (+margin, clamped)
#  - Runs Conventional CIS (SAR ADC) per (size, velocity), captures power, timing, DR/noise, feasibility
#  - Writes CSVs + plots under Use_cases/sweeps_results/
#
# You can choose:
#  - Global contrast (same for all scenes), or
#  - (Advanced) Per-background contrast (uncomment "PER_BACKGROUND" section).

import os, sys, csv, re, io, math
from contextlib import redirect_stdout
from itertools import product
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# --- Make sure repo root is on path (same style as your Use_cases scripts) ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(REPO_ROOT) == 'Use_cases':
    sys.path.append(os.path.dirname(REPO_ROOT))
else:
    sys.path.append(REPO_ROOT)

# --- Import scene model helpers (Ramaa) ---
try:
    from visualcomputing import (
        object_sizes, velocities, safety_factor,
        compute_fps_min, compute_min_snr,  # SNR helper used for bits-from-scene
        # backgrounds  # <- only needed if you enable per-background mode
    )
except Exception as e:
    # Fallback defaults so script still runs
    print('[WARN] Could not import visualcomputing.py:', e)
    object_sizes = [25, 50, 100]
    velocities   = [10, 50, 100, 200, 500]
    safety_factor = 10
    def compute_fps_min(obj_speed, obj_size, safety=10):
        return round((obj_speed / obj_size) * safety, 2)
    def compute_min_snr(obj_contrast: float, noise_floor: float = 1.0) -> float:
        # basic fallback if visualcomputing missing
        return round(20.0 * math.log10(obj_contrast / max(noise_floor, 1e-9)), 2)

# --- Convert SNR (dB) to ENOB ---
def enob_from_snr_db(snr_db: float) -> float:
    return (snr_db - 1.76) / 6.02

# ==========================
#   CONFIGURATION
# ==========================
# Fixed CIS configuration (Conventional CIS + SAR ADC)
PIXEL_BINNING   = [1, 1]
INPUT_CLK_FREQ  = 20e6
PGA_DC_GAIN     = 17.38
INPUT_PIXEL_MAP = [
    [0.2, 1.75, 1.75],
    [0.2, 1.75, 1.75],
    [0.2, 1.75, 1.75],
    [0.2, 1.75, 1.75],
    [0.2, 1.75, 1.75],
]
ROWS, COLS = 480, 640

# --- ADC bits selection ---
BITS_FROM_SCENE = True         # <— set False to revert to a fixed ADC_BITS
ADC_BITS_FIXED  = 10           # used when BITS_FROM_SCENE=False
BITS_MARGIN     = 1            # +1 bit guard band above ENOB
BITS_CLAMP      = (8, 12)      # clamp between these (adjust if your model supports wider range)

# Option A (default): GLOBAL contrast (same for all scenes)
GLOBAL_CONTRAST = 16.0         # e.g., 16:1 contrast; change as needed
NOISE_FLOOR     = 1.0          # keep 1.0 unless you have a different scene noise baseline

# --- Option B: PER_BACKGROUND contrast (advanced) ---
# PER_BACKGROUND = True
# BACKGROUND_CONTRAST = {
#     "low_texture": 16.0,     # higher effective contrast
#     "high_texture": 8.0,     # lower effective contrast -> more bits
# }
# BACKGROUNDS = list(BACKGROUND_CONTRAST.keys())

# --- Outputs under Use_cases/sweeps_results ---
OUT_DIR = os.path.join(REPO_ROOT if os.path.basename(REPO_ROOT) == 'Use_cases'
                       else os.path.join(REPO_ROOT, 'Use_cases'), 'sweeps_results')
os.makedirs(OUT_DIR, exist_ok=True)
CSV_SUMMARY = os.path.join(OUT_DIR, 'cis_all_scenes_summary.csv')
CSV_FULL    = os.path.join(OUT_DIR, 'cis_all_scenes_full_output.csv')
PNG_POWER_LINES = os.path.join(OUT_DIR, 'cis_power_vs_velocity_by_object_size.png')
PNG_FPS_LINES   = os.path.join(OUT_DIR, 'cis_required_fps_vs_velocity.png')
PNG_HEATMAP     = os.path.join(OUT_DIR, 'cis_power_heatmap_mW.png')
PNG_FEAS        = os.path.join(OUT_DIR, 'cis_feasibility_heatmap.png')

# ==========================
#   Helpers
# ==========================
def _find_float(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None

def parse_metrics(out: str) -> Dict[str, Any]:
    return {
        'effective_pixels':      _find_float(r"Num of Effective pixels\s*:\s*([0-9.]+)", out),
        'frame_time_ms':         _find_float(r"Frame Time\s*:\s*([0-9.]+)\s*ms", out),
        'exposure_time_us':      _find_float(r"Exposure Time\s*:\s*([0-9.]+)\s*us", out),
        'readout_time_us':       _find_float(r"Readout Time\s*:\s*([0-9.]+)\s*us", out),
        'io_time_us':            _find_float(r"I/O Time\s*:\s*([0-9.]+)\s*us", out),
        'adc_time_us':           _find_float(r"ADC Time\s*:\s*([0-9.]+)\s*us", out),
        'idle_time_us':          _find_float(r"Idle Time\s*:\s*([0-9.]+)\s*us", out),
        'frame_rate_hz_printed': _find_float(r"Frame Rate\s*:\s*([0-9.]+)\s*HZ", out),
        'max_frame_rate_hz':     _find_float(r"Max Frame Rate\s*:\s*([0-9.]+)\s*Hz", out),
        'system_total_power_mW_printed': _find_float(r"System Total Power\s*:\s*([0-9.]+)\s*mW", out),
        'dynamic_range_dB':      _find_float(r"0th Color,\s*0th pixel size\s*:\s*([0-9.]+)\s*dB", out),
        'total_noise_e':         _find_float(r"Total Noise\s*:\s*([0-9.]+)\s*e-", out),
        'fps_clipped_warning':   1 if 'Input frame rate is too high' in out else 0,
    }

def bits_from_contrast(contrast: float, noise_floor: float = NOISE_FLOOR,
                       margin_bits: int = BITS_MARGIN, clamp=BITS_CLAMP) -> int:
    """Compute ADC bits from contrast via SNR -> ENOB, apply +margin and clamp."""
    snr_db = compute_min_snr(contrast, noise_floor)
    enob   = enob_from_snr_db(snr_db)
    bits   = int(math.ceil(enob + margin_bits))
    bits   = max(clamp[0], min(bits, clamp[1]))
    return bits

# ==========================
#   Run all scenes
# ==========================
summary_rows = []
full_rows    = []

adc_label = (f"ADC variable (min bits from SNR, clamp {BITS_CLAMP[0]}–{BITS_CLAMP[1]})"
             if BITS_FROM_SCENE else f"ADC {ADC_BITS_FIXED}-bit (fixed)")

print('\n=== ModuCIS one-click runner: ALL scene-model pairs ===\n')
print(f'Object sizes: {object_sizes}')
print(f'Velocities  : {velocities}')
print(f'Rule: fps_min = (velocity/object_size) * safety_factor, safety_factor={safety_factor}')
print(f'ADC policy: {adc_label}')
if BITS_FROM_SCENE:
    print(f'Global contrast = {GLOBAL_CONTRAST} (change in file if needed)\n')
else:
    print()

# If you want per-background logic, uncomment the PER_BACKGROUND block above and expand loops accordingly.
# Here we keep the default: GLOBAL contrast for all scenes.

# --- CIS model entry point (same import used in your sweeps) ---
from Top_10_22_CNN_optical import CIS_Array

for obj_px, vel_px_s in product(object_sizes, velocities):
    # Required FPS from scene
    required_fps = compute_fps_min(vel_px_s, obj_px, safety_factor)

    # Bits policy
    if BITS_FROM_SCENE:
        adc_bits_this_scene = bits_from_contrast(GLOBAL_CONTRAST, NOISE_FLOOR)
    else:
        adc_bits_this_scene = ADC_BITS_FIXED

    # Run CIS at the required FPS and computed bits
    buf = io.StringIO()
    with redirect_stdout(buf):
        sensor = CIS_Array(
            feature_size_nm=65,
            analog_V_dd=2.8,
            digital_V_dd=0.8,
            input_clk_freq=INPUT_CLK_FREQ,
            photodiode_type=0,
            CFA_type=0,
            input_pixel_map=INPUT_PIXEL_MAP,
            pd_E=10,
            PD_saturation_level=0.6,
            num_rows=ROWS,
            num_cols=COLS,
            Pixel_type=0,
            pixel_binning_map=PIXEL_BINNING,
            num_PD_per_tap=1,
            num_of_tap=1,
            num_of_unused_tap=0,
            frame_rate=float(required_fps),
            max_subframe_rates=1,
            subframe_rates=1,
            exposure_time=0,
            IO_type=1,
            MUX_type=0,
            PGA_type=1,
            CDS_type=0,
            CIS_type=0,    # Conventional CIS
            ADC_type=1,    # SAR ADC
            adc_resolution=int(adc_bits_this_scene),
            PGA_DC_gain=PGA_DC_GAIN,
            num_mux_input=1,
            CDS_amp_gain=8,
            bias_voltage=0.8,
            comparator_bias_voltage=0.8,
            ADC_input_clk_freq=22e6,
            if_PLL=1,
            if_time_generator=1,
            PLL_output_frequency=44e6,
            additional_latency=0,
            CNN_kernel=3,
        )
        print('Total Power (sensor.system_total_power):', sensor.system_total_power)

    out_text = buf.getvalue()
    met = parse_metrics(out_text)

    # Prefer attribute power when available
    try:
        power_W = float(sensor.system_total_power)
        power_mW_attr = power_W * 1000.0
    except Exception:
        power_W = None
        power_mW_attr = None

    # If attribute missing, fall back to printed total power
    if power_mW_attr is None:
        p_mW_printed = met.get('system_total_power_mW_printed')
        power_mW = p_mW_printed
    else:
        power_mW = power_mW_attr

    max_fr = met.get('max_frame_rate_hz')
    feasible = (max_fr is None) or (required_fps <= max_fr)

    # Console quick log (now includes bits)
    print(f"obj={obj_px:>3}px  vel={vel_px_s:>4} px/s  reqFPS={required_fps:>6}  "
          f"bits={adc_bits_this_scene:>2}  Power≈{(power_mW if power_mW is not None else float('nan')):.2f} mW  "
          f"MaxFR={max_fr}  Feasible={'YES' if feasible else 'NO'}")

    # Append summary row
    summary_rows.append({
        'object_size_px': obj_px,
        'velocity_px_s': vel_px_s,
        'required_fps': required_fps,
        'rows': ROWS, 'cols': COLS,
        'adc_bits_used': int(adc_bits_this_scene),
        'power_mW': power_mW,
        'power_W': (power_W if power_W is not None else (power_mW/1000.0 if power_mW is not None else None)),
        'max_frame_rate_hz': max_fr,
        'feasible': int(1 if feasible else 0),
        'frame_time_ms': met.get('frame_time_ms'),
        'readout_time_us': met.get('readout_time_us'),
        'adc_time_us': met.get('adc_time_us'),
        'io_time_us': met.get('io_time_us'),
        'dynamic_range_dB': met.get('dynamic_range_dB'),
        'total_noise_e': met.get('total_noise_e'),
        'fps_clipped_warning': met.get('fps_clipped_warning'),
    })

    # Append full output row (raw text)
    full_rows.append({
        'object_size_px': obj_px,
        'velocity_px_s': vel_px_s,
        'required_fps': required_fps,
        'rows': ROWS, 'cols': COLS,
        'adc_bits_used': int(adc_bits_this_scene),
        'full_output': out_text,
    })

# --- Write CSVs ---
fieldnames = [
    'object_size_px','velocity_px_s','required_fps','rows','cols',
    'adc_bits_used','power_W','power_mW',
    'max_frame_rate_hz','feasible','frame_time_ms','readout_time_us','adc_time_us','io_time_us',
    'dynamic_range_dB','total_noise_e','fps_clipped_warning'
]
with open(CSV_SUMMARY, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)
print('Saved summary CSV:', CSV_SUMMARY)

with open(CSV_FULL, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'object_size_px','velocity_px_s','required_fps','rows','cols',
        'adc_bits_used','full_output'
    ])
    w.writeheader()
    for r in full_rows:
        w.writerow(r)
print('Saved raw-output CSV:', CSV_FULL)

# --- Build grids for plots ---
# Unique sorted axes
sizes = sorted(set(r['object_size_px'] for r in summary_rows))
vels  = sorted(set(r['velocity_px_s'] for r in summary_rows))
size_to_idx = {s:i for i,s in enumerate(sizes)}
vel_to_idx  = {v:i for i,v in enumerate(vels)}

P = np.full((len(sizes), len(vels)), np.nan)
F = np.zeros((len(sizes), len(vels)), dtype=int)  # feasibility
for r in summary_rows:
    i = size_to_idx[r['object_size_px']]
    j = vel_to_idx[r['velocity_px_s']]
    P[i,j] = (r['power_mW'] if r['power_mW'] is not None else np.nan)
    F[i,j] = r['feasible']

# --- Plot 1: Power vs Velocity (one line per object size) ---
plt.figure(figsize=(8,5))
for i, s in enumerate(sizes):
    y = [P[i, vel_to_idx[v]] for v in vels]
    plt.plot(vels, y, marker='o', label=f'Object {s}px')
plt.xlabel('Object Velocity (pixels/second)')
plt.ylabel('CIS Total Power (mW)')
plt.title(f'ModuCIS (Conventional CIS): Power vs Velocity @ {ROWS}x{COLS}, ADC={adc_label}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PNG_POWER_LINES, dpi=200)
print('Saved plot:', PNG_POWER_LINES)

# --- Plot 2: Required FPS vs Velocity (for reference) ---
plt.figure(figsize=(8,5))
for s in sizes:
    y = [compute_fps_min(v, s, safety_factor) for v in vels]
    plt.plot(vels, y, marker='o', label=f'Object {s}px')
plt.xlabel('Object Velocity (pixels/second)')
plt.ylabel('Required FPS (fps_min)')
plt.title('Scene model: Required FPS vs Velocity (per object size)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PNG_FPS_LINES, dpi=200)
print('Saved plot:', PNG_FPS_LINES)

# --- Plot 3: Power heatmap (mW) ---
plt.figure(figsize=(9,5))
im = plt.imshow(P, aspect='auto', interpolation='nearest', cmap='viridis',
                extent=[min(vels), max(vels), min(sizes), max(sizes)], origin='lower')
plt.colorbar(im, label='Power (mW)')
plt.xlabel('Velocity (px/s)')
plt.ylabel('Object size (px)')
plt.title(f'CIS Power Heatmap (mW) @ {ROWS}x{COLS}, ADC={adc_label}')
plt.tight_layout()
plt.savefig(PNG_HEATMAP, dpi=200)
print('Saved plot:', PNG_HEATMAP)

# --- Plot 4: Feasibility heatmap (1=OK, 0=exceeds max FPS) ---
plt.figure(figsize=(9,5))
im2 = plt.imshow(F, aspect='auto', interpolation='nearest', cmap='Greens', vmin=0, vmax=1,
                 extent=[min(vels), max(vels), min(sizes), max(sizes)], origin='lower')
plt.colorbar(im2, label='Feasible (1=yes, 0=no)')
plt.xlabel('Velocity (px/s)')
plt.ylabel('Object size (px)')
plt.title('CIS Feasibility vs Required FPS and Max Frame Rate')
plt.tight_layout()
plt.savefig(PNG_FEAS, dpi=200)
print('Saved plot:', PNG_FEAS)

print('\nAll done. Check outputs in:', OUT_DIR)
