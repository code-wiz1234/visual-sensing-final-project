# final_cis_withbg_dependent_contrast.py
# One-click ModuCIS runner (Conventional CIS + SAR ADC) across ALL scene-model pairs
# with per-background **required SNR** → ENOB → bits (+margin, clamped).
#
# Why this version?
# - High-texture (cluttered) scenes are harder → higher required SNR → more ADC bits.
# - Low-texture (clean) scenes are easier → lower required SNR → fewer bits.
# - Keeps a realistic CIS window: 8–12 bits (adjustable).
#
# Usage (from repo root or from Use_cases/):
#   python3 "final_cis_withbg_dependent_contrast.py"
#
# Outputs:
#   Use_cases/sweeps_results/
#     - cis_all_scenes_summary_with_bg.csv (background, adc_bits_used, required_fps, power, feasibility, timing)
#     - cis_all_scenes_full_output_with_bg.csv (raw model printouts)
#     - PNG plots per background: power vs velocity, required FPS vs velocity, heatmaps

import os, sys, csv, re, io, math
from contextlib import redirect_stdout
from itertools import product
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Repo path setup (same as other Use_cases scripts)
# -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(HERE) == 'Use_cases':
    REPO_ROOT = os.path.dirname(HERE)
    sys.path.append(REPO_ROOT)
else:
    REPO_ROOT = HERE
    sys.path.append(REPO_ROOT)

# -----------------------------
# Import scene model (Ramaa)
# Expects: object_sizes, velocities, backgrounds, safety_factor, compute_fps_min
# -----------------------------
try:
    from visualcomputing import (
        object_sizes, velocities, backgrounds, safety_factor,
        compute_fps_min
    )
except Exception as e:
    print('[WARN] Could not import visualcomputing.py. Using fallbacks:', e)
    object_sizes = [25, 50, 100]
    velocities   = [10, 50, 100, 200, 500]
    backgrounds  = {"low_texture": 0.05, "high_texture": 0.40}
    safety_factor = 10
    def compute_fps_min(v, s, safety=10): return round((v/s) * safety, 2)

def enob_from_snr_db(snr_db: float) -> float:
    """Ideal ENOB from SNR (dB)."""
    return (snr_db - 1.76) / 6.02

# -----------------------------
# CIS configuration (Conventional CIS + SAR ADC)
# -----------------------------
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

# -----------------------------
# Bits-from-scene policy (TASK-DRIVEN)
# Required SNR per background → ENOB → bits (+margin), clamped to 8..12
# Tweak these two values to move bits slightly if needed.
# -----------------------------
NOISE_FLOOR = 1.0    # baseline (used only if you convert contrast→SNR elsewhere)
BITS_MARGIN = 1      # +1 bit guard band above ENOB
BITS_CLAMP  = (8, 12)

# Choose required SNR targets per background (dB)
# Low-texture: easier → lower SNR → ~8 bits
# High-texture: harder → higher SNR → ~9–10 bits
REQUIRED_SNR_DB = {
    "low_texture": 36.0,   # ENOB≈(36-1.76)/6.02 ≈ 5.7 → +1 ≈ 6.7 → clamp→ 8 bits
    "high_texture": 48.0,  # ENOB≈(48-1.76)/6.02 ≈ 7.7 → +1 ≈ 8.7 → clamp→ 9 bits
}
DEFAULT_SNR_DB = 42.0

def bits_from_required_snr(snr_db_req: float,
                           margin_bits: int = BITS_MARGIN,
                           clamp: Tuple[int,int] = BITS_CLAMP) -> int:
    enob_req = enob_from_snr_db(snr_db_req)
    bits     = int(math.ceil(enob_req + margin_bits))
    return max(clamp[0], min(bits, clamp[1]))

# -----------------------------
# Outputs
# -----------------------------
OUT_DIR = os.path.join(
    REPO_ROOT if os.path.basename(HERE) == 'Use_cases' else os.path.join(REPO_ROOT, 'Use_cases'),
    'sweeps_results'
)
os.makedirs(OUT_DIR, exist_ok=True)

CSV_COMBINED = os.path.join(OUT_DIR, 'cis_all_scenes_summary_with_bg.csv')
CSV_FULL     = os.path.join(OUT_DIR, 'cis_all_scenes_full_output_with_bg.csv')

def out_name(base: str, bg: str) -> str:
    stem, ext = os.path.splitext(base)
    return os.path.join(OUT_DIR, f"{stem}_{bg}{ext}")

# -----------------------------
# Helpers to parse printed metrics
# -----------------------------
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

# -----------------------------
# Import CIS model (same as your Use_cases sweeps)
# -----------------------------
from Top_10_22_CNN_optical import CIS_Array

# -----------------------------
# Run per background
# -----------------------------
combined_rows: List[Dict[str,Any]] = []
full_rows:     List[Dict[str,Any]] = []

print("\n=== ModuCIS one-click runner (per-background, required-SNR) ===\n")
print(f"Object sizes (px): {object_sizes}")
print(f"Velocities (px/s): {velocities}")
print(f"Backgrounds: {list(backgrounds.keys())}")
print(f"Rule: fps_min = (velocity / object_size) * safety_factor, safety_factor={safety_factor}")  # scene→CIS
print(f"Bits policy: REQUIRED_SNR_DB[bg] → ENOB + {BITS_MARGIN} bit(s), clamp={BITS_CLAMP}\n")     # scene→CIS

for bg_name in backgrounds.keys():
    # Pick required SNR for this background and convert to bits
    snr_db_req = REQUIRED_SNR_DB.get(bg_name, DEFAULT_SNR_DB)
    bits_for_bg = bits_from_required_snr(snr_db_req)

    print(f"\n-- Background: {bg_name} (required SNR ≈ {snr_db_req:.1f} dB → bits≈{bits_for_bg}) --")

    # Matrices for per-background plots
    sizes_sorted = sorted(object_sizes)
    vels_sorted  = sorted(velocities)
    size_to_idx  = {s:i for i,s in enumerate(sizes_sorted)}
    vel_to_idx   = {v:i for i,v in enumerate(vels_sorted)}

    P = np.full((len(sizes_sorted), len(vels_sorted)), np.nan)  # power (mW)
    F = np.zeros((len(sizes_sorted), len(vels_sorted)), dtype=int)  # feasibility (1/0)

    for obj_px, vel_px_s in product(sizes_sorted, vels_sorted):
        # Required FPS from scene model
        required_fps = compute_fps_min(vel_px_s, obj_px, safety_factor)

        # Bits derived for THIS background (constant across scenes under bg assumption)
        adc_bits = bits_for_bg

        # Run the CIS model
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
                adc_resolution=int(adc_bits),
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
            print("Total Power (sensor.system_total_power):", sensor.system_total_power)

        out_text = buf.getvalue()
        met = parse_metrics(out_text)

        # Prefer attribute power (W) if available
        try:
            power_W = float(sensor.system_total_power)
            power_mW_attr = power_W * 1000.0
        except Exception:
            power_W = None
            power_mW_attr = None

        # Fallback to printed total power (mW) if attribute missing
        if power_mW_attr is None:
            p_mW_printed = met.get('system_total_power_mW_printed')
            power_mW = p_mW_printed
        else:
            power_mW = power_mW_attr

        max_fr = met.get('max_frame_rate_hz')
        feasible = (max_fr is None) or (required_fps <= max_fr)

        print(f"bg={bg_name:>12}  obj={obj_px:>3}px  vel={vel_px_s:>4} px/s  "
              f"reqFPS={required_fps:>6}  bits={adc_bits:>2}  "
              f"Power≈{(power_mW if power_mW is not None else float('nan')):.2f} mW  "
              f"MaxFR={max_fr}  Feasible={'YES' if feasible else 'NO'}")

        # Save combined rows
        combined_rows.append({
            'background': bg_name,
            'object_size_px': obj_px,
            'velocity_px_s': vel_px_s,
            'required_fps': required_fps,
            'rows': ROWS, 'cols': COLS,
            'adc_bits_used': int(adc_bits),
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

        # Save raw output
        full_rows.append({
            'background': bg_name,
            'object_size_px': obj_px,
            'velocity_px_s': vel_px_s,
            'required_fps': required_fps,
            'rows': ROWS, 'cols': COLS,
            'adc_bits_used': int(adc_bits),
            'full_output': out_text,
        })

        # Fill matrices for plots
        i = size_to_idx[obj_px]
        j = vel_to_idx[vel_px_s]
        P[i, j] = (power_mW if power_mW is not None else np.nan)
        F[i, j] = 1 if feasible else 0

    # -----------------------------
    # Per-background plots
    # -----------------------------
    # Power vs Velocity (one line per object size)
    png_power_lines = out_name('cis_power_vs_velocity_by_object_size.png', bg_name)
    plt.figure(figsize=(8,5))
    for idx, s in enumerate(sizes_sorted):
        y = [P[idx, vels_sorted.index(v)] for v in vels_sorted]
        plt.plot(vels_sorted, y, marker='o', label=f'Object {s}px')
    plt.xlabel('Object Velocity (pixels/second)')
    plt.ylabel('CIS Total Power (mW)')
    plt.title(f'CIS Power vs Velocity @ {ROWS}x{COLS}, bg={bg_name}, bits≈{bits_for_bg}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_power_lines, dpi=200)
    print('Saved plot:', png_power_lines)

    # Required FPS vs Velocity (reference)
    png_fps_lines = out_name('cis_required_fps_vs_velocity.png', bg_name)
    plt.figure(figsize=(8,5))
    for s in sizes_sorted:
        y = [compute_fps_min(v, s, safety_factor) for v in vels_sorted]
        plt.plot(vels_sorted, y, marker='o', label=f'Object {s}px')
    plt.xlabel('Object Velocity (pixels/second)')
    plt.ylabel('Required FPS (fps_min)')
    plt.title(f'Scene model: Required FPS vs Velocity (bg={bg_name})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_fps_lines, dpi=200)
    print('Saved plot:', png_fps_lines)

    # Power heatmap (mW)
    png_heatmap = out_name('cis_power_heatmap_mW.png', bg_name)
    plt.figure(figsize=(9,5))
    im = plt.imshow(P, aspect='auto', interpolation='nearest', cmap='viridis',
                    extent=[min(vels_sorted), max(vels_sorted),
                            min(sizes_sorted), max(sizes_sorted)],
                    origin='lower')
    plt.colorbar(im, label='Power (mW)')
    plt.xlabel('Velocity (px/s)')
    plt.ylabel('Object size (px)')
    plt.title(f'CIS Power Heatmap (mW) @ {ROWS}x{COLS}, bg={bg_name}')
    plt.tight_layout()
    plt.savefig(png_heatmap, dpi=200)
    print('Saved plot:', png_heatmap)

    # Feasibility heatmap
    png_feas = out_name('cis_feasibility_heatmap.png', bg_name)
    plt.figure(figsize=(9,5))
    im2 = plt.imshow(F, aspect='auto', interpolation='nearest', cmap='Greens', vmin=0, vmax=1,
                     extent=[min(vels_sorted), max(vels_sorted),
                             min(sizes_sorted), max(sizes_sorted)],
                     origin='lower')
    plt.colorbar(im2, label='Feasible (1=yes, 0=no)')
    plt.xlabel('Velocity (px/s)')
    plt.ylabel('Object size (px)')
    plt.title(f'CIS Feasibility vs Required FPS and Max Frame Rate (bg={bg_name})')
    plt.tight_layout()
    plt.savefig(png_feas, dpi=200)
    print('Saved plot:', png_feas)

# -----------------------------
# Write combined CSVs (all backgrounds)
# -----------------------------
fieldnames = [
    'background','object_size_px','velocity_px_s','required_fps','rows','cols',
    'adc_bits_used','power_W','power_mW',
    'max_frame_rate_hz','feasible','frame_time_ms','readout_time_us','adc_time_us','io_time_us',
    'dynamic_range_dB','total_noise_e','fps_clipped_warning'
]

with open(CSV_COMBINED, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in combined_rows:
        w.writerow(r)
print('Saved combined summary CSV:', CSV_COMBINED)

with open(CSV_FULL, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'background','object_size_px','velocity_px_s','required_fps',
        'rows','cols','adc_bits_used','full_output'
    ])
    w.writeheader()
    for r in full_rows:
        w.writerow(r)
print('Saved raw-output CSV:', CSV_FULL)

print('\nAll done. Check outputs in:', OUT_DIR)
