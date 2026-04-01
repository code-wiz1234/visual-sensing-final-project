# final_cis_complete.py
# Harshitha — ModuCIS (Conventional CIS + SAR ADC)
# Team 4: CIS vs DVS Co-Design for Object Tracking
#
# CIS power is CONSTANT — locked to worst-case FPS regardless of velocity.
#
# PART 1  — Full scene analysis (all velocities + object sizes from Ramaa)
#            5 CSVs + 7 plots
#
# PART 1B — Animation scene analysis (velocities_viz only from Ramaa)
#            4 CSVs + 7 plots
#
# PART 2  — Camera vs CIS snapshot animation
#            Interactive window + GIF/MP4 + event rate vs power plot

import os, sys, re, io, math
from contextlib import redirect_stdout
from itertools import product
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Path setup ─────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
CIS_MODEL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, CIS_MODEL)

# ── Import Ramaa's scene model ─────────────────────────────────────────────
try:
    from visualcomputingscene import (
        object_sizes, velocities, velocities_viz,
        backgrounds, safety_factor, false_pos_rate,
        scene_width, scene_height, light_wm2,
        ANIM_BG, ANIM_OBJ_SIZE,
        compute_fps_min, compute_min_snr, compute_event_rate,
        draw_room,
    )
    import matplotlib.patches as patches
    from matplotlib.patches import Circle
    print("[OK] Imported Ramaa's scene model (visualcomputingscene.py)")
    print(f"[OK] Animation velocities : {velocities_viz}")
    print(f"[OK] Animation background : {ANIM_BG}")
    print(f"[OK] Animation object size: {ANIM_OBJ_SIZE}px")
except ImportError as e:
    raise ImportError(
        f"visualcomputingscene.py not found in {HERE}\n"
        f"Make sure Ramaa's file is in the same Use_cases/ folder.\n"
        f"Error: {e}"
    )
except Exception as e:
    print(f"[WARN] Partial import: {e} — using fallbacks")
    try:
        from visualcomputingscene import (
            compute_fps_min, compute_min_snr,
            compute_event_rate, draw_room)
    except Exception:
        def compute_fps_min(v,s,sf=10): return round((v/s)*sf,2)
        def compute_min_snr(c,n=1.0):  return round(20*math.log10(max(c,1e-9)/max(n,1e-9)),2)
        def compute_event_rate(v,s,bg,fp=0.10):
            return round((4*s*v+bg*640*480*0.01*v)*(1+fp),1)
        def draw_room(ax,bg): pass
    import matplotlib.patches as patches
    from matplotlib.patches import Circle
    object_sizes=[25,50,100]; velocities=[10,50,100,200,500,1000,2000]
    velocities_viz=[50,500,2000]; backgrounds={"low_texture":0.05,"high_texture":0.40}
    safety_factor=10; false_pos_rate=0.10
    scene_width=640; scene_height=480; light_wm2=5.0
    ANIM_BG="low_texture"; ANIM_OBJ_SIZE=50

# ── Key parameters from Ramaa ──────────────────────────────────────────────
VELOCITIES_STATIC = sorted(velocities)          # all velocities from her file
VELOCITIES_ANIM   = velocities_viz              # animation velocities only
ANIM_OBJ_SIZE_PX  = ANIM_OBJ_SIZE              # animation object size

# ── Animation constants ────────────────────────────────────────────────────
ANIM_R     = 30
ANIM_Y     = 185
ANIM_X0    = 60
ANIM_STEPS = 60

def _positions(vel, n=ANIM_STEPS):
    ppf = max(4, min(scene_width*0.8/n, vel/VELOCITIES_ANIM[0]*6))
    return [min(ANIM_X0+i*ppf, scene_width-ANIM_R-10) for i in range(n)]

# ── CIS hardware baseline ──────────────────────────────────────────────────
PIXEL_BINNING   = [1, 1]
INPUT_CLK_FREQ  = 20e6
PGA_DC_GAIN     = 17.38
INPUT_PIXEL_MAP = [[0.2, 1.75, 1.75]] * 5
ROWS = int(scene_height)
COLS = int(scene_width)

# ── ADC bits via SNR → ENOB ────────────────────────────────────────────────
REQUIRED_SNR_DB = {"low_texture": 36.0, "high_texture": 48.0}
BITS_MARGIN=1; BITS_CLAMP=(8,12); NOISE_FLOOR=1.0; OBJ_CONTRAST=50.0

def enob_from_snr(snr_db): return (snr_db-1.76)/6.02
def bits_from_snr(snr_db):
    return max(BITS_CLAMP[0], min(BITS_CLAMP[1],
           int(math.ceil(enob_from_snr(snr_db)+BITS_MARGIN))))

# ── Worst-case FPS — LOCKED for all CIS_Array calls ───────────────────────
WORST_VEL      = max(VELOCITIES_STATIC)
WORST_SIZE     = min(object_sizes)
WORST_CASE_FPS = compute_fps_min(WORST_VEL, WORST_SIZE, safety_factor)
print(f"\n[CIS] Worst-case FPS locked to : {WORST_CASE_FPS} fps")
print(f"      (velocity={WORST_VEL} px/s, object={WORST_SIZE} px)")
print(f"[CIS] Power will be CONSTANT across all velocities.\n")

# ── Output directory ───────────────────────────────────────────────────────
OUT_DIR = os.path.join(HERE, 'sweeps_results_final_cis_model')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def _find_float(pat, txt):
    m = re.search(pat, txt, flags=re.MULTILINE)
    return float(m.group(1)) if m else None

def parse_metrics(out):
    return {
        'frame_time_ms':     _find_float(r"Frame Time\s*:\s*([0-9.]+)\s*ms",  out),
        'readout_time_us':   _find_float(r"Readout Time\s*:\s*([0-9.]+)\s*us",out),
        'adc_time_us':       _find_float(r"ADC Time\s*:\s*([0-9.]+)\s*us",    out),
        'io_time_us':        _find_float(r"I/O Time\s*:\s*([0-9.]+)\s*us",    out),
        'max_frame_rate_hz': _find_float(r"Max Frame Rate\s*:\s*([0-9.]+)\s*Hz",   out),
        'power_mW_printed':  _find_float(r"System Total Power\s*:\s*([0-9.]+)\s*mW",out),
        'dynamic_range_dB':  _find_float(r"0th Color,\s*0th pixel size\s*:\s*([0-9.]+)\s*dB",out),
        'total_noise_e':     _find_float(r"Total Noise\s*:\s*([0-9.]+)\s*e-", out),
        'fps_clipped':       1 if 'Input frame rate is too high' in out else 0,
    }

from Top_10_22_CNN_optical import CIS_Array

def run_cis_instance(adc_bits):
    buf = io.StringIO()
    with redirect_stdout(buf):
        sensor = CIS_Array(
            feature_size_nm=65, analog_V_dd=2.8, digital_V_dd=0.8,
            input_clk_freq=INPUT_CLK_FREQ,
            photodiode_type=0, CFA_type=0,
            input_pixel_map=INPUT_PIXEL_MAP,
            pd_E=float(light_wm2), PD_saturation_level=0.6,
            num_rows=ROWS, num_cols=COLS,
            Pixel_type=0, pixel_binning_map=PIXEL_BINNING,
            num_PD_per_tap=1, num_of_tap=1, num_of_unused_tap=0,
            frame_rate=float(WORST_CASE_FPS),   # LOCKED — never changes
            max_subframe_rates=1, subframe_rates=1, exposure_time=0,
            IO_type=1, MUX_type=0, PGA_type=1, CDS_type=0,
            CIS_type=0, ADC_type=1,
            adc_resolution=int(adc_bits),
            PGA_DC_gain=PGA_DC_GAIN, num_mux_input=1,
            CDS_amp_gain=8, bias_voltage=0.8,
            comparator_bias_voltage=0.8,
            ADC_input_clk_freq=22e6, if_PLL=1,
            if_time_generator=1, PLL_output_frequency=44e6,
            additional_latency=0, CNN_kernel=3,
        )
        print("sys_power:", sensor.system_total_power)
    met = parse_metrics(buf.getvalue())
    try:    power_mW = float(sensor.system_total_power) * 1000.0
    except: power_mW = met.get('power_mW_printed') or float('nan')
    return power_mW, met

# ══════════════════════════════════════════════════════════════════════════
#  RUN CIS — once per background, cache result
# ══════════════════════════════════════════════════════════════════════════
print("="*65)
print("Running CIS (once per background, power locked to worst-case FPS)")
print("="*65)

bg_cache: Dict[str, Tuple[float, Dict, int]] = {}
for bg_name in backgrounds:
    snr_req  = REQUIRED_SNR_DB.get(bg_name, 42.0)
    adc_bits = bits_from_snr(snr_req)
    print(f"\n  [{bg_name}]  SNR={snr_req}dB → bits={adc_bits} | FPS={WORST_CASE_FPS}")
    power_mW, met = run_cis_instance(adc_bits)
    bg_cache[bg_name] = (power_mW, met, adc_bits)
    print(f"  [{bg_name}]  Constant power = {power_mW:.4f} mW")

# ══════════════════════════════════════════════════════════════════════════
#  BUILD DATA ROWS
# ══════════════════════════════════════════════════════════════════════════
def make_row(bg_name, obj_px, vel, scene_tag):
    p_mW, met, bits = bg_cache[bg_name]
    lat      = met.get('frame_time_ms') or (1.0/WORST_CASE_FPS)*1000
    feasible = int((met.get('max_frame_rate_hz') is None) or
                   (WORST_CASE_FPS <= (met.get('max_frame_rate_hz') or 1e9)))
    return {
        'scene':            scene_tag,
        'background':       bg_name,
        'object_size_px':   obj_px,
        'velocity_px_s':    vel,
        'required_fps':     compute_fps_min(vel, obj_px, safety_factor),
        'worst_case_fps':   WORST_CASE_FPS,
        'rows': ROWS, 'cols': COLS,
        'adc_bits_used':    bits,
        'power_mW':         round(p_mW, 4),
        'latency_ms':       round(lat,  4),
        'snr_dB':           compute_min_snr(OBJ_CONTRAST, NOISE_FLOOR),
        'dynamic_range_dB': met.get('dynamic_range_dB'),
        'total_noise_e':    met.get('total_noise_e'),
        'max_frame_rate_hz':met.get('max_frame_rate_hz'),
        'feasible':         feasible,
        'fps_clipped':      met.get('fps_clipped'),
        'event_rate_ev_s':  compute_event_rate(
                                vel, obj_px,
                                backgrounds[bg_name], false_pos_rate),
    }

# Full scene rows
static_rows = [make_row(bg, obj, vel, 'static')
               for bg in backgrounds
               for obj, vel in product(sorted(object_sizes), VELOCITIES_STATIC)]

# Animation scene rows — both backgrounds, Ramaa's velocities + object size
anim_rows = [make_row(bg, ANIM_OBJ_SIZE_PX, vel, 'animation')
             for bg in backgrounds
             for vel in VELOCITIES_ANIM]

# ══════════════════════════════════════════════════════════════════════════
#  SAVE CSVs
# ══════════════════════════════════════════════════════════════════════════
import pandas as pd

def save_csv(rows, name):
    path = os.path.join(OUT_DIR, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'[Saved] {path}')

# ── Part 1 CSVs — full scene ───────────────────────────────────────────────
save_csv(static_rows, 'cis_all_scenes_summary.csv')
save_csv([
    {'background': bg, 'adc_bits': bg_cache[bg][2],
     'worst_case_fps': WORST_CASE_FPS,
     'power_mW': round(bg_cache[bg][0], 4),
     'latency_ms': round(bg_cache[bg][1].get('frame_time_ms') or
                         (1/WORST_CASE_FPS)*1000, 4),
     'dynamic_range_dB': bg_cache[bg][1].get('dynamic_range_dB'),
     'total_noise_e':    bg_cache[bg][1].get('total_noise_e')}
    for bg in backgrounds
], 'cis_per_background_summary.csv')

TIME_STEPS        = [0,1,2,3,4,5,6,7,8,9,10,11,12]
VELOCITY_SEQUENCE = [10,50,100,200,500,1000,2000,1000,500,200,100,50,10]
save_csv([
    {'time_s': t, 'velocity_px_s': vel, 'background': bg,
     'power_mW': round(bg_cache[bg][0], 4),
     'latency_ms': round(bg_cache[bg][1].get('frame_time_ms') or
                         (1/WORST_CASE_FPS)*1000, 4),
     'worst_case_fps': WORST_CASE_FPS,
     'note': 'CIS power constant'}
    for bg in backgrounds
    for t, vel in zip(TIME_STEPS, VELOCITY_SEQUENCE)
], 'cis_temporal_variation.csv')

save_csv([
    {'object_size_px': obj, 'velocity_px_s': vel,
     'required_fps': compute_fps_min(vel, obj, safety_factor),
     'worst_case_fps': WORST_CASE_FPS}
    for obj in sorted(object_sizes) for vel in VELOCITIES_STATIC
], 'cis_fps_requirements.csv')

# ── Part 1B CSVs — animation scene ────────────────────────────────────────
save_csv(anim_rows, 'anim_cis_all_scenes_summary.csv')
save_csv([
    {'background': bg, 'adc_bits': bg_cache[bg][2],
     'worst_case_fps': WORST_CASE_FPS,
     'power_mW': round(bg_cache[bg][0], 4),
     'latency_ms': round(bg_cache[bg][1].get('frame_time_ms') or
                         (1/WORST_CASE_FPS)*1000, 4),
     'dynamic_range_dB': bg_cache[bg][1].get('dynamic_range_dB'),
     'total_noise_e':    bg_cache[bg][1].get('total_noise_e')}
    for bg in backgrounds
], 'anim_cis_per_background_summary.csv')
save_csv([
    {'object_size_px': ANIM_OBJ_SIZE_PX, 'velocity_px_s': vel,
     'required_fps': compute_fps_min(vel, ANIM_OBJ_SIZE_PX, safety_factor),
     'worst_case_fps': WORST_CASE_FPS}
    for vel in VELOCITIES_ANIM
], 'anim_cis_fps_requirements.csv')
save_csv([
    {'time_s': t, 'velocity_px_s': vel, 'background': bg,
     'power_mW': round(bg_cache[bg][0], 4),
     'latency_ms': round(bg_cache[bg][1].get('frame_time_ms') or
                         (1/WORST_CASE_FPS)*1000, 4),
     'worst_case_fps': WORST_CASE_FPS,
     'note': 'CIS power constant — animation scene'}
    for bg in backgrounds
    for t, vel in zip([0, 1, 2], VELOCITIES_ANIM)
], 'anim_cis_temporal_variation.csv')

# ══════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════
OBJ_COLORS = {25:'#E55', 50:'#55A', 100:'#282'}
BG_COLORS  = {'low_texture':'#2288CC', 'high_texture':'#EE6622'}
sizes_s    = sorted(object_sizes)

def save_fig(name):
    p = os.path.join(OUT_DIR, name)
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {p}')

def plot_power_vs_vel(df, vels, sizes, prefix, title_tag):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f'CIS Power vs Velocity  [{title_tag}]\n'
                 'Constant — locked to worst-case FPS', fontweight='bold')
    for ax, bg in zip(axes, backgrounds):
        pc = bg_cache[bg][0]
        for obj in sizes:
            sub = df[(df.background==bg)&(df.object_size_px==obj)].sort_values('velocity_px_s')
            ax.plot(sub.velocity_px_s, sub.power_mW,
                    color=OBJ_COLORS.get(obj,'#55A'), lw=2.5,
                    marker='o', label=f'obj={obj}px')
        ax.axhline(y=pc, color='red', ls='--', lw=1.5, label=f'Fixed ({pc:.1f} mW)')
        ax.set(title=f'Background: {bg}', xlabel='Velocity (px/s)',
               ylabel='CIS Power (mW)')
        ax.set_ylim(0, pc*1.3)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    plt.tight_layout(); save_fig(f'{prefix}cis_power_vs_velocity.png')

def plot_eventrate_vs_power(df, sizes, prefix, title_tag):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f'CIS Power vs Scene Event Rate  [{title_tag}]\n'
                 'Power constant regardless of scene activity', fontweight='bold')
    for ax, bg in zip(axes, backgrounds):
        pc = bg_cache[bg][0]
        for obj in sizes:
            sub = df[(df.background==bg)&(df.object_size_px==obj)].sort_values('event_rate_ev_s')
            if sub.empty: continue
            ax.plot(sub.event_rate_ev_s, [pc]*len(sub),
                    color=OBJ_COLORS.get(obj,'#55A'), lw=2.5,
                    marker='o', label=f'obj={obj}px ({pc:.1f} mW)')
        ax.set(title=f'Background: {bg}',
               xlabel='Scene Event Rate (events/s)',
               ylabel='CIS Power (mW)  [constant]')
        ax.set_xscale('log'); ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
        ax.annotate('CIS power does NOT\nchange with scene activity',
                    xy=(0.05,0.85), xycoords='axes fraction', fontsize=9,
                    color='darkred',
                    bbox=dict(boxstyle='round', facecolor='#fff0f0', edgecolor='red'))
    plt.tight_layout(); save_fig(f'{prefix}cis_eventrate_vs_power.png')

def plot_latency(df, sizes, prefix, title_tag):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f'CIS Latency vs Velocity  [{title_tag}]\n'
                 'Constant = 1/worst-case FPS', fontweight='bold')
    for ax, bg in zip(axes, backgrounds):
        lat = bg_cache[bg][1].get('frame_time_ms') or (1/WORST_CASE_FPS)*1000
        for obj in sizes:
            sub = df[(df.background==bg)&(df.object_size_px==obj)].sort_values('velocity_px_s')
            ax.plot(sub.velocity_px_s, sub.latency_ms,
                    color=OBJ_COLORS.get(obj,'#55A'), lw=2.5,
                    marker='^', label=f'obj={obj}px')
        ax.axhline(y=lat, color='purple', ls='--', lw=1.5,
                   label=f'Frame time ({lat:.3f} ms)')
        ax.set(title=f'Background: {bg}', xlabel='Velocity (px/s)',
               ylabel='CIS Latency (ms)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    plt.tight_layout(); save_fig(f'{prefix}cis_latency_vs_velocity.png')

def plot_snr_dr(df, prefix, title_tag):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f'CIS SNR and Dynamic Range  [{title_tag}]', fontweight='bold')
    ax_snr, ax_dr = axes
    for bg in backgrounds:
        sub = df[(df.background==bg)&(df.object_size_px==ANIM_OBJ_SIZE_PX)].sort_values('velocity_px_s')
        if sub.empty: continue
        ax_snr.plot(sub.velocity_px_s, sub.snr_dB,
                    color=BG_COLORS[bg], marker='D', lw=2, label=bg)
    ax_snr.axhline(y=20, color='red', ls='--', lw=1.5, label='Min SNR (20 dB)')
    ax_snr.set(title='SNR vs Velocity', xlabel='Velocity (px/s)', ylabel='SNR (dB)')
    ax_snr.legend(); ax_snr.grid(True, alpha=0.4)
    for bg in backgrounds:
        dr = bg_cache[bg][1].get('dynamic_range_dB')
        if dr:
            sub = df[(df.background==bg)&(df.object_size_px==ANIM_OBJ_SIZE_PX)].sort_values('velocity_px_s')
            if sub.empty: continue
            ax_dr.plot(sub.velocity_px_s, [dr]*len(sub),
                       color=BG_COLORS[bg], marker='s', lw=2,
                       label=f'{bg} ({dr:.1f} dB)')
    ax_dr.set(title='Dynamic Range vs Velocity (constant)',
              xlabel='Velocity (px/s)', ylabel='DR (dB)')
    ax_dr.legend(); ax_dr.grid(True, alpha=0.4)
    plt.tight_layout(); save_fig(f'{prefix}cis_snr_dr_vs_velocity.png')

def plot_heatmaps(df, vels, sizes, prefix, title_tag):
    for bg in backgrounds:
        sub_bg = df[df.background==bg]
        P = np.full((len(sizes), len(vels)), np.nan)
        F = np.zeros_like(P, dtype=int)
        for i, s in enumerate(sizes):
            for j, v in enumerate(vels):
                r = sub_bg[(sub_bg.object_size_px==s)&(sub_bg.velocity_px_s==v)]
                if not r.empty:
                    P[i,j] = r.power_mW.values[0]
                    F[i,j] = r.feasible.values[0]
        for grid, cmap, label, tag in [
            (P,'viridis','CIS Power (mW)','power_heatmap'),
            (F,'Greens', 'Feasible (1=yes 0=no)','feasibility_heatmap')
        ]:
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower',
                           interpolation='nearest',
                           **({'vmin':0,'vmax':1} if 'feasibility' in tag else {}))
            plt.colorbar(im, ax=ax, label=label)
            ax.set_xticks(range(len(vels)))
            ax.set_xticklabels(vels, rotation=45, fontsize=8)
            ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes)
            ax.set(xlabel='Velocity (px/s)', ylabel='Object Size (px)',
                   title=f'CIS {tag.replace("_"," ").title()} [{title_tag}] — {bg}')
            plt.tight_layout()
            save_fig(f'{prefix}cis_{tag}_{bg}.png')

# ══════════════════════════════════════════════════════════════════════════
#  PART 1 PLOTS — Full scene
# ══════════════════════════════════════════════════════════════════════════
df_s = pd.DataFrame(static_rows)
plot_power_vs_vel(df_s,   VELOCITIES_STATIC, sizes_s,        '',      'Full Scene')
plot_eventrate_vs_power(df_s, sizes_s,                        '',      'Full Scene')
plot_latency(df_s,        sizes_s,                            '',      'Full Scene')
plot_snr_dr(df_s,                                             '',      'Full Scene')
plot_heatmaps(df_s,       VELOCITIES_STATIC, sizes_s,        '',      'Full Scene')

# ══════════════════════════════════════════════════════════════════════════
#  PART 1B PLOTS — Animation scene
# ══════════════════════════════════════════════════════════════════════════
df_a = pd.DataFrame(anim_rows)
plot_power_vs_vel(df_a,   VELOCITIES_ANIM, [ANIM_OBJ_SIZE_PX], 'anim_', 'Animation Scene')
plot_eventrate_vs_power(df_a, [ANIM_OBJ_SIZE_PX],               'anim_', 'Animation Scene')
plot_latency(df_a,        [ANIM_OBJ_SIZE_PX],                   'anim_', 'Animation Scene')
plot_snr_dr(df_a,                                                'anim_', 'Animation Scene')
plot_heatmaps(df_a,       VELOCITIES_ANIM, [ANIM_OBJ_SIZE_PX], 'anim_', 'Animation Scene')

# ══════════════════════════════════════════════════════════════════════════
#  PART 2 — Camera vs CIS snapshot animation
#  Both backgrounds side by side + live event rate graph
# ══════════════════════════════════════════════════════════════════════════
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Build frame list
anim_frames = []
for vel in VELOCITIES_ANIM:
    fps_v        = compute_fps_min(vel, ANIM_R*2, safety_factor)
    xs           = _positions(vel)
    fps_scaled   = max(1, min(ANIM_STEPS, int(fps_v/VELOCITIES_ANIM[0]*3)))
    cis_interval = max(1, ANIM_STEPS//fps_scaled)
    cis_x        = xs[0]
    for i, x in enumerate(xs):
        if i % cis_interval == 0: cis_x = x
        anim_frames.append((vel, x, cis_x, fps_v, i, cis_interval))

# Figure: 3 rows
fig2 = plt.figure(figsize=(16, 14))
fig2.patch.set_facecolor("#F0F0F0")
fig2.suptitle(
    "Camera View  vs  CIS Snapshot View\n"
    "CIS freezes between captures — power stays constant as event rate changes",
    fontsize=13, fontweight="bold", color="#111111"
)
ax_cam_low  = fig2.add_subplot(3, 2, 1)
ax_cis_low  = fig2.add_subplot(3, 2, 2)
ax_cam_high = fig2.add_subplot(3, 2, 3)
ax_cis_high = fig2.add_subplot(3, 2, 4)
ax_live     = fig2.add_subplot(3, 1, 3)

for ax, title, bg_key in [
    (ax_cam_low,  "Camera View — Low Texture",   "low_texture"),
    (ax_cis_low,  "CIS Snapshot — Low Texture",  "low_texture"),
    (ax_cam_high, "Camera View — High Texture",  "high_texture"),
    (ax_cis_high, "CIS Snapshot — High Texture", "high_texture"),
]:
    draw_room(ax, bg_key)
    ax.set_xlim(0, scene_width); ax.set_ylim(0, scene_height)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#2C2C3E")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5, color="#111111")

def make_sphere_set(ax):
    sph = Circle((ANIM_X0,ANIM_Y), ANIM_R, color="#2255AA", zorder=5)
    shi = Circle((ANIM_X0-ANIM_R*.33, ANIM_Y+ANIM_R*.33),
                 ANIM_R*.30, color="white", alpha=0.5, zorder=6)
    shd = patches.Ellipse((ANIM_X0,105), ANIM_R*1.4, ANIM_R*.22,
                           facecolor="black", alpha=0.18, zorder=3)
    trl = [Circle((ANIM_X0,ANIM_Y), ANIM_R,
                   color="#4488DD", alpha=0.12*(i+1), zorder=3)
           for i in range(3)]
    for p in [shd, sph, shi, *trl]: ax.add_patch(p)
    return sph, shi, shd, trl, [ANIM_X0]*3

def make_cis_sphere(ax):
    sph = Circle((ANIM_X0,ANIM_Y), ANIM_R, color="#EE5522", zorder=5)
    shi = Circle((ANIM_X0-ANIM_R*.33, ANIM_Y+ANIM_R*.33),
                 ANIM_R*.30, color="white", alpha=0.5, zorder=6)
    shd = patches.Ellipse((ANIM_X0,105), ANIM_R*1.4, ANIM_R*.22,
                           facecolor="black", alpha=0.18, zorder=3)
    fl  = patches.Rectangle((0,0), scene_width, scene_height,
                             facecolor='white', alpha=0.0, zorder=8)
    for p in [shd, sph, shi, fl]: ax.add_patch(p)
    return sph, shi, shd, fl

c_sph_l,c_shi_l,c_shd_l,trails_l,xh_l = make_sphere_set(ax_cam_low)
c_sph_h,c_shi_h,c_shd_h,trails_h,xh_h = make_sphere_set(ax_cam_high)
k_sph_l,k_shi_l,k_shd_l,flash_l       = make_cis_sphere(ax_cis_low)
k_sph_h,k_shi_h,k_shd_h,flash_h       = make_cis_sphere(ax_cis_high)

t_cl = ax_cam_low.text( 10,scene_height-18,'',color='white',fontsize=8,fontweight='bold',zorder=10)
t_kl = ax_cis_low.text( 10,scene_height-18,'',color='white',fontsize=8,fontweight='bold',zorder=10)
t_fl = ax_cis_low.text( 10,scene_height-34,'',color='#AAFFAA',fontsize=7,zorder=10)
t_ch = ax_cam_high.text(10,scene_height-18,'',color='white',fontsize=8,fontweight='bold',zorder=10)
t_kh = ax_cis_high.text(10,scene_height-18,'',color='white',fontsize=8,fontweight='bold',zorder=10)
t_fh = ax_cis_high.text(10,scene_height-34,'',color='#AAFFAA',fontsize=7,zorder=10)

# Live graph
all_evt_low  = [compute_event_rate(vel, ANIM_OBJ_SIZE_PX,
                backgrounds['low_texture'],  false_pos_rate)
                for vel,*_ in anim_frames]
all_evt_high = [compute_event_rate(vel, ANIM_OBJ_SIZE_PX,
                backgrounds['high_texture'], false_pos_rate)
                for vel,*_ in anim_frames]
max_evt = max(max(all_evt_low), max(all_evt_high)) * 1.15
p_low   = bg_cache['low_texture'][0]
p_high  = bg_cache['high_texture'][0]
p_max   = max(p_low, p_high)

ax_live.set_xlim(0, len(anim_frames))
ax_live.set_ylim(0, p_max*1.4)
ax_live.set_xlabel(
    f'Animation Frame  ({" → ".join(str(v) for v in VELOCITIES_ANIM)} px/s)',
    fontsize=10, color="#111111")
ax_live.set_ylabel('CIS Power (mW)', fontsize=10, color="#111111")
ax_live.set_title('Live: Scene Event Rate changes — CIS Power stays CONSTANT',
                  fontsize=10, fontweight='bold', color="#111111")
ax_live.set_facecolor("#FAFAFA"); ax_live.grid(True, alpha=0.4)
ax_live.axhline(y=p_low,  color='#CC2222', lw=2.5, ls='--',
                label=f'CIS power low_texture  ({p_low:.1f} mW)')
ax_live.axhline(y=p_high, color='#994400', lw=2.5, ls='-.',
                label=f'CIS power high_texture ({p_high:.1f} mW)')
for xv in [ANIM_STEPS, ANIM_STEPS*2]:
    ax_live.axvline(x=xv, color='#888888', lw=1, ls=':')
for xi, lbl in zip([ANIM_STEPS*0.5, ANIM_STEPS*1.5, ANIM_STEPS*2.5],
                   [f'{VELOCITIES_ANIM[0]} px/s',
                    f'{VELOCITIES_ANIM[1]} px/s',
                    f'{VELOCITIES_ANIM[2]} px/s']):
    ax_live.text(xi, p_max*1.28, lbl, ha='center', fontsize=9, color='#333333')

ax_evt = ax_live.twinx()
ax_evt.set_ylim(0, max_evt*1.4)
ax_evt.set_ylabel('Scene Event Rate (events/s)', fontsize=10, color='#2255AA')
ax_evt.tick_params(axis='y', labelcolor='#2255AA')
evt_line_l, = ax_evt.plot([],[], color='#2255AA', lw=2, label='Event rate — low texture')
evt_line_h, = ax_evt.plot([],[], color='#EE6622', lw=2, ls='--', label='Event rate — high texture')
evt_dot_l,  = ax_evt.plot([],[], 'o', color='#2255AA', ms=7, zorder=6)
evt_dot_h,  = ax_evt.plot([],[], 'o', color='#EE6622', ms=7, zorder=6)
lines1,labs1 = ax_live.get_legend_handles_labels()
lines2,labs2 = ax_evt.get_legend_handles_labels()
ax_live.legend(lines1+lines2, labs1+labs2, loc='upper left', fontsize=8,
               facecolor='white', edgecolor='#cccccc')

live_x, live_yl, live_yh = [], [], []

def _update(fi):
    vel, xc, xk, fps_v, pi, ci = anim_frames[fi]
    el = all_evt_low[fi]; eh = all_evt_high[fi]

    def move_cam(sph, shi, shd, trl, xh, txt):
        xh.append(xc); xh.pop(0)
        for tp, hx in zip(trl, xh): tp.center = (hx, ANIM_Y)
        sph.center = (xc, ANIM_Y)
        shi.center = (xc-ANIM_R*.33, ANIM_Y+ANIM_R*.33)
        shd.set_center((xc, 105))
        txt.set_text(f"vel = {vel} px/s")

    def move_cis(sph, shi, shd, fl, txt, fps_txt):
        sph.center = (xk, ANIM_Y)
        shi.center = (xk-ANIM_R*.33, ANIM_Y+ANIM_R*.33)
        shd.set_center((xk, 105))
        txt.set_text(f"vel = {vel} px/s")
        fps_txt.set_text(f"min FPS={fps_v:.0f} | every ~{ci} frames")
        fl.set_alpha(0.30 if pi % ci == 0 else 0.0)

    move_cam(c_sph_l,c_shi_l,c_shd_l,trails_l,xh_l,t_cl)
    move_cam(c_sph_h,c_shi_h,c_shd_h,trails_h,xh_h,t_ch)
    move_cis(k_sph_l,k_shi_l,k_shd_l,flash_l,t_kl,t_fl)
    move_cis(k_sph_h,k_shi_h,k_shd_h,flash_h,t_kh,t_fh)

    live_x.append(fi); live_yl.append(el); live_yh.append(eh)
    evt_line_l.set_data(live_x, live_yl)
    evt_line_h.set_data(live_x, live_yh)
    evt_dot_l.set_data([fi], [el])
    evt_dot_h.set_data([fi], [eh])

    return (*trails_l, c_sph_l, c_shi_l, c_shd_l,
            *trails_h, c_sph_h, c_shi_h, c_shd_h,
            k_sph_l,k_shi_l,k_shd_l,flash_l,
            k_sph_h,k_shi_h,k_shd_h,flash_h,
            t_cl,t_kl,t_fl,t_ch,t_kh,t_fh,
            evt_line_l,evt_line_h,evt_dot_l,evt_dot_h)

plt.tight_layout(rect=[0, 0, 1, 0.95])
anim2 = FuncAnimation(fig2, _update, frames=len(anim_frames),
                      interval=60, blit=True)

try:
    from matplotlib.animation import FFMpegWriter
    mp4 = os.path.join(OUT_DIR, 'anim_cis_camera_vs_snapshot_animation.mp4')
    anim2.save(mp4, writer=FFMpegWriter(fps=16, bitrate=2000),
               savefig_kwargs={'facecolor':'#F0F0F0'})
    print(f'[Saved] {mp4}')
except Exception:
    gif = os.path.join(OUT_DIR, 'anim_cis_camera_vs_snapshot_animation.gif')
    anim2.save(gif, writer='pillow', fps=16,
               savefig_kwargs={'facecolor':'#F0F0F0'})
    print(f'[Saved] {gif}')

plt.show()   # interactive window

# ── Animation event rate vs power plot ────────────────────────────────────
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(
    f'CIS Power vs Scene Event Rate  [Animation Scene]\n'
    f'CIS power flat as velocity changes: {VELOCITIES_ANIM} px/s',
    fontweight='bold')
for ax, bg in zip(axes3, backgrounds):
    pc  = bg_cache[bg][0]
    sub = df_a[df_a.background==bg].sort_values('event_rate_ev_s')
    if sub.empty:
        ax.set_title(f'Background: {bg} (no data)'); continue
    ax.plot(sub.event_rate_ev_s, [pc]*len(sub),
            color='red', lw=2.5, ls='--', label=f'CIS power ({pc:.1f} mW)')
    for _, row in sub.iterrows():
        ax.scatter(row.event_rate_ev_s, pc, s=120, zorder=5, color='#55A')
        ax.annotate(f"{int(row.velocity_px_s)} px/s",
                    (row.event_rate_ev_s, pc),
                    textcoords='offset points', xytext=(0,12),
                    ha='center', fontsize=9, color='#333')
    ax.set(title=f'Background: {bg}',
           xlabel='Scene Event Rate (events/s)',
           ylabel='CIS Power (mW)  [constant]')
    ax.set_xscale('log'); ax.set_ylim(0, pc*1.4)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    ax.annotate('CIS power does NOT change\nwith scene event rate',
                xy=(0.05,0.85), xycoords='axes fraction', fontsize=9,
                color='darkred',
                bbox=dict(boxstyle='round', facecolor='#fff0f0', edgecolor='red'))
plt.tight_layout()
save_fig('anim_cis_animation_eventrate_vs_power.png')

# ══════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('CIS CONSTANT POWER SUMMARY')
print('='*65)
print(f'  Worst-case FPS : {WORST_CASE_FPS} fps'
      f'  (vel={WORST_VEL}px/s, obj={WORST_SIZE}px)')
print(f'  Resolution     : {ROWS} x {COLS}')
for bg in backgrounds:
    p_mW, met, bits = bg_cache[bg]
    lat = met.get('frame_time_ms') or (1/WORST_CASE_FPS)*1000
    print(f'\n  [{bg}]')
    print(f'    ADC bits         : {bits}')
    print(f'    Power (constant) : {p_mW:.4f} mW')
    print(f'    Frame latency    : {lat:.4f} ms')
    print(f'    Dynamic range    : {met.get("dynamic_range_dB","N/A")} dB')
print('\n' + '='*65)
print(f'All outputs in: {OUT_DIR}')
print('\n  FULL SCENE (Part 1):')
print('    CSVs: cis_all_scenes_summary.csv')
print('          cis_per_background_summary.csv')
print('          cis_temporal_variation.csv')
print('          cis_fps_requirements.csv')
print('    PNGs: cis_power_vs_velocity.png')
print('          cis_eventrate_vs_power.png')
print('          cis_latency_vs_velocity.png')
print('          cis_snr_dr_vs_velocity.png')
print('          cis_power_heatmap_[bg].png       (x2)')
print('          cis_feasibility_heatmap_[bg].png (x2)')
print('\n  ANIMATION SCENE (Part 1B + Part 2):')
print('    CSVs: anim_cis_all_scenes_summary.csv')
print('          anim_cis_per_background_summary.csv')
print('          anim_cis_temporal_variation.csv')
print('          anim_cis_fps_requirements.csv')
print('    PNGs: anim_cis_power_vs_velocity.png')
print('          anim_cis_eventrate_vs_power.png')
print('          anim_cis_latency_vs_velocity.png')
print('          anim_cis_snr_dr_vs_velocity.png')
print('          anim_cis_power_heatmap_[bg].png       (x2)')
print('          anim_cis_feasibility_heatmap_[bg].png (x2)')
print('          anim_cis_animation_eventrate_vs_power.png')
print('    ANIM: anim_cis_camera_vs_snapshot_animation.gif / .mp4')
