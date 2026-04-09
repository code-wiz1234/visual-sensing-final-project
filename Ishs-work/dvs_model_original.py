import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# DVS_model updated
# --- Import Ramaa's scene model ---

SCENE_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', 'Remaas-work'
))
sys.path.insert(0, SCENE_MODEL_PATH)

from visualcomputing import (  # type: ignore
    object_sizes, velocities, backgrounds, false_pos_rate,
    scene_width, scene_height, compute_event_rate,
)

# --- DVS Circuit Parameters (prof's model, defaults match Lichtsteiner 128×128 @ 3.3V) ---
# These replace the old hardcoded P_STATIC_MW / E_PER_EVENT_NJ constants.
# Static power and energy-per-event are now computed from first principles.

DVS_VDD      = 3.3        # V  — supply voltage
DVS_C1       = 467e-15    # F  — integration capacitor
DVS_C2       =  24e-15    # F  — feedback capacitor
DVS_CGATE    =  15e-15    # F  — gate capacitance
DVS_C3       =  32e-15    # F  — drain capacitance
DVS_CBUS     =   2.0e-12  # F  — bus capacitance
DVS_CARB     =  20e-15    # F  — arbiter capacitance
DVS_NUM_BITS =  15        # —  — address bits
DVS_IAFE     =  50e-9     # A  — analog front-end bias current per pixel
DVS_IOTHER   =   1e-3     # A  — global static current (logic, I/O)
DVS_ILEAK    =   2e-12    # A  — total leakage current per pixel
DVS_TREF     =   0.1e-6   # s  — refractory period

# False event fractions (applied to the standard event rate)
DVS_FP_RATE  = 0.10   # false positives — fire but shouldn't (waste power)
DVS_FN_RATE  = 0.05   # false negatives — should fire but don't (save power)

# --- Per-pixel Frame Model Parameters ---
# Synthetic frame intensities (0–255 linear scale)
SPHERE_INTENSITY  =  60.0   # sphere pixel intensity — darker than background
BG_INTENSITY_BASE = 180.0   # background base intensity
BG_TEXTURE_SCALE  =  50.0   # static texture amplitude (× bg_density)

# Frame-to-frame temporal noise: simulates photon shot noise, illumination
# flicker, and minor background motion. Scales with bg_density because cluttered
# scenes have more reflective surfaces and micro-motion.
TEMPORAL_NOISE_BASE =  2.0   # std in low-texture scenes
TEMPORAL_NOISE_BG   = 20.0   # extra std added per unit of bg_density

# --- Temporal Variation Sequence ---
# Velocity range is extended beyond the scene model (up to 2000 px/s) to make the
# dynamic power variation clearly visible — at 500 px/s the change is too small
# relative to static power to see on a plot. Higher velocities show the full range.
TIME_STEPS        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]   # seconds
VELOCITY_SEQUENCE = [10, 50, 100, 200, 500, 1000, 2000, 1000, 500, 200, 100, 50, 10]  # px/s
TEMPORAL_OBJ_SIZE = 50

# --- Contrast Threshold Sweep (event_rate ∝ 1/theta, from Lichtsteiner eq. 5) ---

BASELINE_THETA = 0.20  # theta Ramaa's event rate formula is calibrated to
THRESHOLDS = {
    "low_threshold":  0.10,   # 2x more events
    "med_threshold":  0.20,   # baseline
    "high_threshold": 0.40,   # 2x fewer events
}

# --- Pixel Firing Breakdown ---

def compute_pixel_breakdown(obj_size: int, bg_density: float, velocity: float = 100) -> dict:
    n_total      = scene_width * scene_height                    # all pixels
    n_obj_active = 4 * obj_size                                  # object edge pixels (constant)
    # Background pixels scale with velocity — faster motion sweeps more background
    # edges into the active zone per unit time. Normalized to baseline velocity of 100 px/s.
    n_bg_active  = int(bg_density * n_total * 0.01 * (velocity / 100))
    n_active     = min(n_obj_active + n_bg_active, n_total)
    n_silent     = n_total - n_active
    return {
        'n_total_pixels':   n_total,
        'n_active_pixels':  n_active,
        'n_silent_pixels':  n_silent,
        'active_fraction':  round(n_active / n_total, 5),
    }

# --- Per-pixel Frame Model ---

def generate_frame(obj_size: int, cx: float, bg_density: float,
                   noise_seed: int) -> np.ndarray:
    """
    Render a synthetic grayscale frame (scene_height × scene_width, float32).

    The background texture is fixed (seed=42) — identical in both frames so that
    only pixels affected by sphere motion produce a d_ln_I signal.
    Per-frame temporal noise (noise_seed) simulates photon shot noise and
    illumination flicker; its amplitude scales with bg_density so cluttered
    scenes produce more background events, matching physical intuition.
    """
    H, W = scene_height, scene_width

    # Static background texture — same seed → same texture in both frames
    bg_rng = np.random.RandomState(42)
    frame  = BG_INTENSITY_BASE + bg_rng.randn(H, W) * (bg_density * BG_TEXTURE_SCALE)

    # Per-frame temporal noise — independent each call
    t_rng        = np.random.RandomState(noise_seed)
    temporal_std = TEMPORAL_NOISE_BASE + bg_density * TEMPORAL_NOISE_BG
    frame        = frame + t_rng.randn(H, W) * temporal_std

    frame = np.clip(frame, 1.0, 255.0).astype(np.float32)

    # Draw sphere (filled circle, horizontally centred on cx, vertically centred)
    cy = H // 2
    r  = obj_size / 2.0
    Y, X = np.ogrid[:H, :W]
    frame[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = SPHERE_INTENSITY

    return frame


def _render_frame_arr(bg: np.ndarray, temporal_std: float, cx: float,
                      r: float, Y: np.ndarray, X: np.ndarray,
                      seed: int) -> np.ndarray:
    """
    Internal helper: add per-frame temporal noise to a pre-computed background
    array and draw the sphere. Faster than generate_frame() for animation loops
    because bg is pre-built once outside the loop.
    """
    t_rng = np.random.RandomState(seed)
    frame = bg + t_rng.randn(*bg.shape) * temporal_std
    frame = np.clip(frame, 1.0, 255.0).astype(np.float32)
    cy    = bg.shape[0] // 2
    frame[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = SPHERE_INTENSITY
    return frame


def compute_animation_event_rates(obj_size: int, bg_density: float,
                                   thetas: list) -> dict:
    """
    Sweep the sphere left-to-right across the full scene (1px per step),
    compare every consecutive frame pair, and accumulate per-pixel d_ln_I
    statistics for each threshold.

    All thetas are evaluated in a single pass — d_ln_I is computed once per
    frame transition and thresholded at each theta value. This is more accurate
    than a single centre-frame snapshot because it captures how fired pixel
    counts vary as the sphere passes in front of different background regions.

    Returns: {theta: {'fired_mean', 'fired_min', 'fired_max'}}
    fired_mean is used for event_rate = fired_mean × velocity (events/sec).
    Does NOT depend on velocity — caller applies the velocity scaling.
    """
    H, W = scene_height, scene_width
    r    = obj_size / 2.0

    # Pre-build static background (fixed seed — same in every frame)
    bg_rng = np.random.RandomState(42)
    bg     = BG_INTENSITY_BASE + bg_rng.randn(H, W) * (bg_density * BG_TEXTURE_SCALE)
    bg     = np.clip(bg, 1.0, 255.0).astype(np.float32)

    temporal_std = TEMPORAL_NOISE_BASE + bg_density * TEMPORAL_NOISE_BG

    # Sphere sweeps from its leading edge at the left wall to its trailing edge
    # at the right wall — fully visible throughout (no partial-sphere frames)
    cx_positions = np.arange(r, W - r + 1, 1.0)
    n_steps      = len(cx_positions) - 1

    Y, X    = np.ogrid[:H, :W]
    counts  = {theta: [] for theta in thetas}

    prev     = _render_frame_arr(bg, temporal_std, cx_positions[0], r, Y, X, seed=0)
    log_prev = np.log(prev.astype(np.float64))

    for i in range(1, n_steps + 1):
        curr     = _render_frame_arr(bg, temporal_std, cx_positions[i], r, Y, X, seed=i)
        log_curr = np.log(curr.astype(np.float64))
        d_ln_I   = np.abs(log_curr - log_prev)

        for theta in thetas:
            counts[theta].append(int(np.sum(d_ln_I >= theta)))

        log_prev = log_curr   # reuse log for next step — saves one np.log call

    results = {}
    for theta in thetas:
        c = counts[theta]
        results[theta] = {
            'fired_mean': round(float(np.mean(c)), 1),
            'fired_min':  int(min(c)),
            'fired_max':  int(max(c)),
        }
    return results


# --- Circuit-level Power Helpers (from prof's model) ---

def compute_energy_per_event(theta: float) -> float:
    """
    Energy per event in nJ, derived from circuit parameters.

    e_afe = theta² × C2 × (1 + C2/C1)          — analog front-end charge
    e_dig = Vdd² × C_sw_dig                      — digital switching energy
    C_sw_dig accounts for gate caps, arbiter tree, and bus load.
    Higher theta → more charge stored before firing → higher e_afe.
    """
    import math
    e_afe    = (theta ** 2) * DVS_C2 * (1.0 + DVS_C2 / DVS_C1)
    # Arbiter tree depth = log2(W) + log2(H) for a rectangular W×H array
    c_sw_dig = ((DVS_CGATE + DVS_C3)
                + ((math.log2(scene_width) + math.log2(scene_height)) * DVS_CARB)
                + (0.5  * DVS_NUM_BITS * DVS_CBUS))
    e_dig    = (DVS_VDD ** 2) * c_sw_dig
    return (e_afe + e_dig) * 1e9   # J → nJ


def compute_static_power_mw() -> float:
    """
    Static power in mW — always-on bias currents regardless of scene activity.
    Scales with pixel count: n_pixels × (I_afe + I_leak) + I_other (global).
    """
    n_pixels  = scene_width * scene_height   # 640×480 = Ramaa's scene
    i_static  = n_pixels * (DVS_IAFE + DVS_ILEAK) + DVS_IOTHER
    return DVS_VDD * i_static * 1e3   # W → mW


# --- Core Model ---

def compute_dvs_power(event_rate: float, obj_size: int = 50,
                      velocity: float = 100, theta: float = BASELINE_THETA) -> dict:
    """
    Compute DVS power from first principles using prof's circuit model.

    Energy per event is theta-dependent (higher theta → more analog charge).
    False positives add wasted dynamic power; false negatives save power.
    Refractory period caps the maximum array event rate.
    """
    e_per_event_nJ = compute_energy_per_event(theta)
    p_static_mW    = compute_static_power_mw()

    # Refractory cap: each pixel can fire at most once per T_ref
    max_rate_array    = (scene_width * scene_height) / DVS_TREF
    event_rate_capped = min(event_rate, max_rate_array)

    # FP/FN breakdown (from prof's model)
    r_fp    = DVS_FP_RATE * event_rate_capped   # spurious events — waste power
    r_fn    = DVS_FN_RATE * event_rate_capped   # missed events  — save power
    r_valid = event_rate_capped - r_fn

    e_J = e_per_event_nJ * 1e-9
    p_dyn_valid_mW = r_valid * e_J * 1e3
    p_dyn_fp_mW    = r_fp    * e_J * 1e3
    p_dyn_fn_mW    = r_fn    * e_J * 1e3
    p_total_mW     = p_static_mW + p_dyn_valid_mW + p_dyn_fp_mW

    # Tracking feasibility: enough events to detect the object edge once per crossing
    min_detection_rate = 4 * obj_size * (velocity / obj_size)
    dvs_can_track      = int(event_rate_capped >= min_detection_rate)

    return {
        'event_rate_eff':    round(event_rate_capped, 1),
        'e_per_event_nJ':    round(e_per_event_nJ, 4),
        'power_static_mW':   round(p_static_mW, 3),
        'power_dyn_valid_mW': round(p_dyn_valid_mW, 3),
        'power_dyn_fp_mW':   round(p_dyn_fp_mW, 3),
        'power_dyn_fn_mW':   round(p_dyn_fn_mW, 3),
        'power_total_mW':    round(p_total_mW, 3),
        'dvs_can_track':     dvs_can_track,
    }

# --- Run All Scenes ---

def run_all_scenes() -> pd.DataFrame:
    rows       = []
    theta_list = list(THRESHOLDS.values())

    for bg_name, bg_density in backgrounds.items():
        for obj_size in object_sizes:
            # One full animation sweep covers all thetas and all velocities.
            # fired_mean does not depend on velocity — velocity is applied below.
            print(f'  animating: bg={bg_name}, obj={obj_size}px ...', flush=True)
            anim = compute_animation_event_rates(obj_size, bg_density, theta_list)

            for vel in velocities:
                analytical_rate = compute_event_rate(vel, obj_size, bg_density, false_pos_rate)
                px_info         = compute_pixel_breakdown(obj_size, bg_density, vel)

                for th_name, theta in THRESHOLDS.items():
                    a          = anim[theta]
                    event_rate = a['fired_mean'] * vel   # events/sec

                    power_info = compute_dvs_power(
                        event_rate, obj_size=obj_size, velocity=vel, theta=theta,
                    )
                    rows.append({
                        'object_size_px':        obj_size,
                        'velocity_px_s':         vel,
                        'background':            bg_name,
                        'threshold_name':        th_name,
                        'theta':                 theta,
                        'event_rate_analytical': analytical_rate,
                        'fired_pixels_mean':     a['fired_mean'],
                        'fired_pixels_min':      a['fired_min'],
                        'fired_pixels_max':      a['fired_max'],
                        'event_rate_pixelwise':  round(event_rate, 1),
                        **power_info,
                        **px_info,
                    })
    return pd.DataFrame(rows)

def plot_pixel_breakdown(df, out_dir):
    # Worst-case velocity: DVS fires the most events here, making the active fraction
    # as large as possible — this is the hardest scenario for DVS power efficiency.
    # Using max(VELOCITY_SEQUENCE) to capture the extended range (2000 px/s).
    worst_vel = max(VELOCITY_SEQUENCE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (bg_name, bg_df) in zip(axes, df.groupby('background')):
        # Pixel breakdown computed directly since worst_vel may exceed scene model velocities
        bg_density = backgrounds[bg_name]
        px = compute_pixel_breakdown(50, bg_density, worst_vel)
        n_active   = px['n_active_pixels']
        n_silent   = px['n_silent_pixels']
        n_total    = px['n_total_pixels']
        pct_active = px['active_fraction'] * 100
        pct_silent = 100 - pct_active

        wedges, _ = ax.pie(
            [n_silent, n_active],
            colors=['steelblue', 'tomato'],
            startangle=90,
            wedgeprops=dict(width=0.45),   # donut width
        )
        # centre text — big percentage
        ax.text(0, 0.12, f'{pct_silent:.2f}%', ha='center', va='center',
                fontsize=16, fontweight='bold', color='steelblue')
        ax.text(0, -0.18, 'silent', ha='center', va='center',
                fontsize=10, color='steelblue')

        # actual numbers below the donut
        ax.text(0, -0.72,
                f'Silent:  {n_silent:,} px\nActive:  {n_active:,} px\nTotal:   {n_total:,} px',
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#cccccc'))

        ax.set_title(f'{bg_name}\n(velocity = {worst_vel} px/s)', fontsize=11, fontweight='bold')
        ax.legend(wedges, [f'Silent ({pct_silent:.2f}%)', f'Active ({pct_active:.3f}%)'],
                  loc='upper right', fontsize=8)

    plt.suptitle('DVS: Active vs Silent Pixels at Worst-Case Velocity\n'
                 'Static power covers ALL pixels - active fraction is tiny',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_pixel_breakdown.png'), dpi=200)
    plt.close()

# --- Temporal Variation ---

def run_temporal_variation() -> pd.DataFrame:
    rows = []
    for t, vel in zip(TIME_STEPS, VELOCITY_SEQUENCE):
        for bg_name, bg_density in backgrounds.items():
            raw_rate = compute_event_rate(vel, TEMPORAL_OBJ_SIZE, bg_density, false_pos_rate)
            rows.append({
                'time_s':         t,
                'velocity_px_s':  vel,
                'background':     bg_name,
                'event_rate_raw': raw_rate,
                **compute_dvs_power(raw_rate),
            })
    return pd.DataFrame(rows)

# CIS must be configured for the fastest object it will ever see — it can't adapt frame-by-frame
# So it runs at worst-case FPS (and power) even during slow moments.
def get_cis_worst_case_power_mw() -> float | None:
    cis_csv = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'Harshithas-work',
        'Spring-2026-ModuCIS-modeling-main', 'ModuCIS.-CIS-modeling-main', 'ModuCIS.-CIS-modeling-main',
        'CIS_Model', 'Use_cases', 'sweeps_results_final_cis_model',
        'cis_all_scenes_summary.csv'
    ))
    if not os.path.exists(cis_csv):
        return None
    import pandas as _pd
    cis_df = _pd.read_csv(cis_csv)
    # Harshita's CSV now covers the full velocity range (up to 2000 px/s)
    worst_vel = max(velocities)
    row = cis_df[(cis_df['velocity_px_s'] == worst_vel) &
                 (cis_df['object_size_px'] == TEMPORAL_OBJ_SIZE)]
    return float(row['power_mW'].values[0]) if not row.empty else None

def plot_temporal_variation(df_temporal, out_dir):
    cis_mw = get_cis_worst_case_power_mw()
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=False)

    ymin_by_bg = {'high_texture': 53.92, 'low_texture': 53.98}
    ymax_by_bg = {'high_texture': 54.6,  'low_texture': 54.14}
    for ax, (bg_name, bg_df) in zip(axes, df_temporal.groupby('background')):
        bg_df      = bg_df.sort_values('time_s')
        p_static   = bg_df['power_static_mW'].iloc[0]
        p_total    = bg_df['power_total_mW']

        ax.plot(bg_df['time_s'], p_total,
                marker='o', color='#2255AA', linewidth=2.5, label='DVS total power')
        # Fill only the dynamic component above the static floor
        ax.fill_between(bg_df['time_s'], p_static, p_total,
                        alpha=0.30, color='#2255AA', label='Dynamic component')
        ax.axhline(y=p_static, color='steelblue', linewidth=1.5,
                   linestyle=':', label='Static floor (53.99 mW)')

        for _, r in bg_df.iterrows():
            ax.annotate(f"{int(r['velocity_px_s'])}px/s",
                        (r['time_s'], r['power_total_mW']),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=7, ha='center')

        if cis_mw is not None:
            dvs_max = p_total.max()
            if cis_mw <= dvs_max * 1.3:
                ax.axhline(y=cis_mw, color='orange', linewidth=2, linestyle='--',
                           label=f'CIS power (fixed @ {cis_mw:.1f} mW)')
            else:
                ax.annotate(f'CIS power: {cis_mw:.1f} mW (above chart)',
                            xy=(0.02, 0.95), xycoords='axes fraction',
                            fontsize=8, color='black',
                            bbox=dict(boxstyle='round', facecolor='#fff8e7', edgecolor='orange'))

        ax.annotate('Note: y-axis range differs between high- and low-texture plots',
                    xy=(0.05, 0.02), xycoords='axes fraction',
                    fontsize=9, color='black', style='italic')

        ax.set_ylim(bottom=ymin_by_bg.get(bg_name), top=ymax_by_bg.get(bg_name))
        ax.set(title=f'DVS Power Over Time ({bg_name})',
               xlabel='Time (s)', ylabel='Power (mW)')
        ax.legend(fontsize=8); ax.grid(True)

    plt.suptitle('DVS: Temporal Variation (power tracks scene activity)\n'
                 'Shaded area = dynamic power above static floor',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_temporal_variation.png'), dpi=200)
    plt.close()

# --- Plots ---

def plot_pixelwise_vs_analytical(df, out_dir):
    """
    Side-by-side comparison of pixelwise vs analytical event rates.
    Shows how the per-pixel d_ln_I model differs from the analytical formula
    and how threshold theta controls how many pixels fire.
    Only med_threshold (theta=0.20) shown to keep the plot readable; object size=50.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    sub = df[(df['threshold_name'] == 'med_threshold') &
             (df['object_size_px'] == 50)]

    for ax, (bg_name, bg_df) in zip(axes, sub.groupby('background')):
        bg_df = bg_df.sort_values('velocity_px_s')
        ax.plot(bg_df['velocity_px_s'], bg_df['event_rate_analytical'],
                marker='o', linestyle='--', color='gray',  label='Analytical (Ramaa formula)')
        ax.plot(bg_df['velocity_px_s'], bg_df['event_rate_pixelwise'],
                marker='s', linestyle='-',  color='#2255AA', label='Pixelwise |d_ln_I| ≥ θ')
        ax.set(title=f'Event Rate: Pixelwise vs Analytical\n({bg_name}, θ=0.20)',
               xlabel='Velocity (px/s)', ylabel='Event Rate (events/sec)')
        ax.legend(); ax.grid(True)

    plt.suptitle('Per-pixel d_ln_I Model vs Analytical Formula\n'
                 'Pixelwise rate reflects actual pixel threshold crossings',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_pixelwise_vs_analytical.png'), dpi=200)
    plt.close()


def plot_power_vs_velocity(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=False)
    for ax, (bg_name, bg_df) in zip(axes, df.groupby('background')):
        for th_name, th_df in bg_df.groupby('threshold_name'):
            sub = th_df[th_df['object_size_px'] == 50].sort_values('velocity_px_s')
            ax.plot(sub['velocity_px_s'], sub['power_total_mW'], marker='o', label=th_name)
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(bottom=y_lo - (y_hi - y_lo) * 0.1)
        if bg_name == 'low_texture':
            ax.annotate('Threshold lines overlap: static power\ndominates, dynamic variation < 0.04 mW',
                        xy=(0.98, 0.05), xycoords='axes fraction', ha='right',
                        fontsize=8, color='black',
                        bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc'))
        ax.set(title=f'DVS Power vs Velocity ({bg_name})',
               xlabel='Velocity (px/s)', ylabel='DVS Total Power (mW)')
        ax.legend(); ax.grid(True)
    plt.suptitle('DVS: Power vs Velocity', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_vs_velocity.png'), dpi=200)
    plt.close()

def plot_power_vs_background(df, out_dir):
    # low_threshold (θ=0.10) is the most sensitive/reliable DVS setting — equivalent to
    # CIS using a 10x safety factor. Using this as the fair comparison point.
    fig, ax = plt.subplots(figsize=(8, 5))
    for bg_name, bg_df in df[df['threshold_name'] == 'low_threshold'].groupby('background'):
        sub = bg_df[bg_df['object_size_px'] == 50].sort_values('velocity_px_s')
        ax.plot(sub['velocity_px_s'], sub['power_total_mW'], marker='s', label=bg_name)
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(bottom=y_lo - (y_hi - y_lo) * 0.1)
    ax.set(title='DVS Power: Low vs High Texture (low_threshold is a conservative operating point)',
           xlabel='Velocity (px/s)', ylabel='DVS Total Power (mW)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_vs_background.png'), dpi=200)
    plt.close()

def plot_power_vs_threshold(df, out_dir):
    ymax_by_bg = {'high_texture': None, 'low_texture': 54.045}
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=False)
    for ax, (bg_name, bg_df) in zip(axes, df.groupby('background')):
        for vel, vel_df in bg_df[bg_df['object_size_px'] == 50].groupby('velocity_px_s'):
            sub = vel_df.sort_values('theta')
            ax.plot(sub['theta'], sub['power_total_mW'], marker='o', label=f'{vel} px/s')
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(bottom=y_lo - (y_hi - y_lo) * 0.1, top=ymax_by_bg.get(bg_name))
        ax.annotate('Note: y-axis range differs between high- and low-texture plots',
                    xy=(0.05, 0.02), xycoords='axes fraction',
                    fontsize=9, color='black', style='italic')
        ax.set(title=f'DVS Power vs Threshold ({bg_name})',
               xlabel='Threshold θ', ylabel='DVS Total Power (mW)')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True)
    plt.suptitle('DVS: Power vs Threshold', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_vs_threshold.png'), dpi=200)
    plt.close()

def plot_static_vs_dynamic(df, out_dir):
    sub = df[(df['threshold_name'] == 'high_threshold') &
             (df['background'] == 'low_texture') &
             (df['object_size_px'] == 50)].sort_values('velocity_px_s')
    x = sub['velocity_px_s'].astype(str)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Top: full stacked bar — static + valid dynamic + false positive waste
    ax1.bar(x, sub['power_static_mW'],    label='Static',           color='steelblue')
    ax1.bar(x, sub['power_dyn_valid_mW'], label='Dynamic (valid)',   color='tomato',
            bottom=sub['power_static_mW'])
    ax1.bar(x, sub['power_dyn_fp_mW'],    label='Dynamic (false +)', color='#e74c3c',
            bottom=sub['power_static_mW'] + sub['power_dyn_valid_mW'], hatch='//')
    ax1.set(title='DVS Power Breakdown: Full View',
            xlabel='Velocity (px/s)', ylabel='Power (mW)')
    ax1.legend(); ax1.grid(True, axis='y')

    # Bottom: dynamic only (valid + FP) — zoomed to show velocity effect
    p_dyn_total = sub['power_dyn_valid_mW'] + sub['power_dyn_fp_mW']
    ax2.bar(x, sub['power_dyn_valid_mW'], label='Dynamic (valid)',   color='tomato')
    ax2.bar(x, sub['power_dyn_fp_mW'],    label='Dynamic (false +)', color='#e74c3c',
            bottom=sub['power_dyn_valid_mW'], hatch='//')
    for i, (_, r) in enumerate(sub.iterrows()):
        total_dyn = r['power_dyn_valid_mW'] + r['power_dyn_fp_mW']
        ax2.text(i, total_dyn + 0.0005, f"{total_dyn:.4f}", ha='center', fontsize=8)
    ax2.set(title='Dynamic Power Only: Zoomed (valid + false positive waste)',
            xlabel='Velocity (px/s)', ylabel='Dynamic Power (mW)')
    ax2.legend(); ax2.grid(True, axis='y')

    plt.suptitle('DVS Power Breakdown (high_threshold, low_texture)\n'
                 'Static dominates — dynamic includes false positive waste',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_breakdown.png'), dpi=200)
    plt.close()

# --- Main ---

if __name__ == '__main__':
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dvs_results')
    os.makedirs(OUT_DIR, exist_ok=True)

    df = run_all_scenes()
    df.to_csv(os.path.join(OUT_DIR, 'dvs_all_scenes_summary.csv'), index=False)

    print(f'P_static = {compute_static_power_mw():.2f} mW | '
          f'E_per_event (θ=0.20) = {compute_energy_per_event(BASELINE_THETA):.4f} nJ')
    # print(df[(df['threshold_name'] == 'med_threshold') &
    # (df['background'] == 'low_texture')].to_string(index=False))

    plot_pixelwise_vs_analytical(df, OUT_DIR)
    plot_power_vs_velocity(df, OUT_DIR)
    plot_power_vs_background(df, OUT_DIR)
    plot_power_vs_threshold(df, OUT_DIR)
    plot_static_vs_dynamic(df, OUT_DIR)
    plot_pixel_breakdown(df, OUT_DIR)

    df_temporal = run_temporal_variation()
    df_temporal.to_csv(os.path.join(OUT_DIR, 'dvs_temporal_variation.csv'), index=False)
    plot_temporal_variation(df_temporal, OUT_DIR)

    print('Done. Results in:', OUT_DIR)