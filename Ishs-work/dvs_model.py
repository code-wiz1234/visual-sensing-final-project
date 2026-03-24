import os
import sys
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

# --- DVS Hardware Parameters (Lichtsteiner 2008, 128x128 sensor @ 3.3V) ---

# Static power: logic (0.3mA) + bias generators (5.5mA)
P_STATIC_MW    = (0.3 + 5.5) * 3.3                             # 19.14 mW

# Dynamic energy per event: core current (1.5mA) / max event rate (1M ev/s)
E_PER_EVENT_NJ = (1.5 * 3.3 * 1e-3) / 1_000_000 * 1e9          # 4.95 nJ

# Refractory cap scaled from 128x128 to 640x480
REFRACTORY_CAP = 1_000_000 * (scene_width * scene_height) / (128 * 128)  # 18.75M ev/s

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

# --- Core Model ---

def compute_dvs_power(event_rate_raw: float, theta: float = BASELINE_THETA) -> dict:
    event_rate_scaled = event_rate_raw * (BASELINE_THETA / theta)
    saturated         = event_rate_scaled > REFRACTORY_CAP
    event_rate_eff    = min(event_rate_scaled, REFRACTORY_CAP)
    power_dynamic_mW  = (event_rate_eff * E_PER_EVENT_NJ * 1e-9) * 1e3
    power_total_mW    = P_STATIC_MW + power_dynamic_mW
    return {
        'event_rate_scaled':  round(event_rate_scaled, 1),
        'event_rate_eff':     round(event_rate_eff, 1),
        'power_static_mW':    round(P_STATIC_MW, 3),
        'power_dynamic_mW':   round(power_dynamic_mW, 3),
        'power_total_mW':     round(power_total_mW, 3),
        'saturated':          int(saturated),
    }

# --- Run All Scenes ---

def run_all_scenes() -> pd.DataFrame:
    rows = []
    for bg_name, bg_density in backgrounds.items():
        for obj_size in object_sizes:
            for vel in velocities:
                raw_rate  = compute_event_rate(vel, obj_size, bg_density, false_pos_rate)
                px_info   = compute_pixel_breakdown(obj_size, bg_density, vel)
                for th_name, theta in THRESHOLDS.items():
                    rows.append({
                        'object_size_px': obj_size,
                        'velocity_px_s':  vel,
                        'background':     bg_name,
                        'threshold_name': th_name,
                        'theta':          theta,
                        'event_rate_raw': raw_rate,
                        **compute_dvs_power(raw_rate, theta),
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
                 'Static power covers ALL pixels — active fraction is tiny',
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
        'ModuCIS.-CIS-modeling-main', 'ModuCIS.-CIS-modeling-main',
        'CIS_Model', 'Use_cases', 'sweeps_results_final_cis_model',
        'cis_all_scenes_summary.csv'
    ))
    if not os.path.exists(cis_csv):
        return None
    import pandas as _pd
    cis_df = _pd.read_csv(cis_csv)
    # Cap at 500 — Harshita's CSV only covers up to 500 px/s regardless of scene model range - currently
    worst_vel = min(max(velocities), 500)
    row = cis_df[(cis_df['velocity_px_s'] == worst_vel) &
                 (cis_df['object_size_px'] == TEMPORAL_OBJ_SIZE)]
    return float(row['power_mW'].values[0]) if not row.empty else None

def plot_temporal_variation(df_temporal, out_dir):
    cis_mw = get_cis_worst_case_power_mw()
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=True)

    for ax, (bg_name, bg_df) in zip(axes, df_temporal.groupby('background')):
        bg_df = bg_df.sort_values('time_s')
        ax.plot(bg_df['time_s'], bg_df['power_total_mW'],
                marker='o', color='#2255AA', linewidth=2.5, label='DVS power')
        ax.fill_between(bg_df['time_s'], bg_df['power_total_mW'],
                        alpha=0.15, color='#2255AA')
        for _, r in bg_df.iterrows():
            ax.annotate(f"{int(r['velocity_px_s'])}px/s",
                        (r['time_s'], r['power_total_mW']),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=7, ha='center')
        if cis_mw is not None:
            dvs_max = bg_df['power_total_mW'].max()
            if cis_mw <= dvs_max * 1.3:
                # CIS is in range — draw as horizontal line
                ax.axhline(y=cis_mw, color='orange', linewidth=2, linestyle='--',
                           label=f'CIS power (fixed @ {cis_mw:.1f} mW)')
            else:
                # CIS is far above DVS range — annotate so DVS peaks stay visible
                ax.annotate(f'CIS power: {cis_mw:.1f} mW (above chart)',
                            xy=(0.02, 0.95), xycoords='axes fraction',
                            fontsize=8, color='black',
                            bbox=dict(boxstyle='round', facecolor='#fff8e7', edgecolor='orange'))
        # Start y-axis just below static floor so dynamic variation is visible
        ax.set_ylim(bottom=P_STATIC_MW * 0.97)
        ax.set(title=f'DVS Power Over Time ({bg_name})',
               xlabel='Time (s)', ylabel='Power (mW)')
        ax.legend(); ax.grid(True)

    plt.suptitle('DVS: Temporal Variation — Power Tracks Scene Activity\n'
                 '(CIS locked to worst-case FPS regardless of velocity)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_temporal_variation.png'), dpi=200)
    plt.close()

# --- Plots ---

def plot_power_vs_velocity(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=True)
    for ax, (bg_name, bg_df) in zip(axes, df.groupby('background')):
        for th_name, th_df in bg_df.groupby('threshold_name'):
            sub = th_df[th_df['object_size_px'] == 50].sort_values('velocity_px_s')
            ax.plot(sub['velocity_px_s'], sub['power_total_mW'], marker='o', label=th_name)
        ax.set(title=f'DVS Power vs Velocity ({bg_name})',
               xlabel='Velocity (px/s)', ylabel='DVS Total Power (mW)')
        ax.legend(); ax.grid(True)
    plt.suptitle('DVS: Power vs Velocity', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_vs_velocity.png'), dpi=200)
    plt.close()

def plot_power_vs_background(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for bg_name, bg_df in df[df['threshold_name'] == 'med_threshold'].groupby('background'):
        sub = bg_df[bg_df['object_size_px'] == 50].sort_values('velocity_px_s')
        ax.plot(sub['velocity_px_s'], sub['power_total_mW'], marker='s', label=bg_name)
    ax.set(title='DVS Power: Low vs High Texture (med_threshold)',
           xlabel='Velocity (px/s)', ylabel='DVS Total Power (mW)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_power_vs_background.png'), dpi=200)
    plt.close()

def plot_power_vs_threshold(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=True)
    for ax, (bg_name, bg_df) in zip(axes, df.groupby('background')):
        for vel, vel_df in bg_df[bg_df['object_size_px'] == 50].groupby('velocity_px_s'):
            sub = vel_df.sort_values('theta')
            ax.plot(sub['theta'], sub['power_total_mW'], marker='o', label=f'{vel} px/s')
        ax.set(title=f'DVS Power vs Threshold ({bg_name})',
               xlabel='Threshold θ', ylabel='DVS Total Power (mW)')
        ax.legend(title='Velocity'); ax.grid(True)
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

    # Left: full stacked bar — shows static dominance
    ax1.bar(x, sub['power_static_mW'], label='Static (mW)', color='steelblue')
    ax1.bar(x, sub['power_dynamic_mW'], label='Dynamic (mW)', color='tomato',
            bottom=sub['power_static_mW'])
    ax1.set(title='DVS Power Breakdown — Full View',
            xlabel='Velocity (px/s)', ylabel='Power (mW)')
    ax1.legend(); ax1.grid(True, axis='y')

    # Right: dynamic power only — zoomed in to show how it changes with velocity
    ax2.bar(x, sub['power_dynamic_mW'], color='tomato', label='Dynamic (mW)')
    for i, (_, r) in enumerate(sub.iterrows()):
        ax2.text(i, r['power_dynamic_mW'] + 0.002,
                 f"{r['power_dynamic_mW']:.3f}", ha='center', fontsize=8)
    ax2.set(title='Dynamic Power Only — Zoomed',
            xlabel='Velocity (px/s)', ylabel='Dynamic Power (mW)')
    ax2.legend(); ax2.grid(True, axis='y')

    plt.suptitle('DVS Power Breakdown (high_threshold, low_texture)\n'
                 'Static power dominates — dynamic is barely visible at full scale',
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

    print(f'P_static = {P_STATIC_MW:.2f} mW | E_per_event = {E_PER_EVENT_NJ:.3f} nJ | '
          f'Refractory cap = {REFRACTORY_CAP/1e6:.2f}M ev/s')
    # print(df[(df['threshold_name'] == 'med_threshold') &
    #          (df['background'] == 'low_texture')].to_string(index=False))

    plot_power_vs_velocity(df, OUT_DIR)
    plot_power_vs_background(df, OUT_DIR)
    plot_power_vs_threshold(df, OUT_DIR)
    plot_static_vs_dynamic(df, OUT_DIR)
    plot_pixel_breakdown(df, OUT_DIR)

    df_temporal = run_temporal_variation()
    df_temporal.to_csv(os.path.join(OUT_DIR, 'dvs_temporal_variation.csv'), index=False)
    plot_temporal_variation(df_temporal, OUT_DIR)

    print('Done. Results in:', OUT_DIR)