import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- Import Ramaa's scene model ---

SCENE_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', 'Harshitas-work-S26-ModuCIS-modeling-main',
    'ModuCIS.-CIS-modeling-main', 'ModuCIS.-CIS-modeling-main',
    'CIS_Model', 'Use_cases'
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
E_PER_EVENT_NJ = (1.5 * 3.3 * 1e-3) / 1_000_000 * 1e9         # 4.95 nJ

# Refractory cap scaled from 128x128 to 640x480
REFRACTORY_CAP = 1_000_000 * (scene_width * scene_height) / (128 * 128)  # 18.75M ev/s

# --- Contrast Threshold Sweep (event_rate ∝ 1/theta, from Lichtsteiner eq. 5) ---

BASELINE_THETA = 0.20  # theta Ramaa's event rate formula is calibrated to
THRESHOLDS = {
    "low_threshold":  0.10,   # 2x more events
    "med_threshold":  0.20,   # baseline
    "high_threshold": 0.40,   # 2x fewer events
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
                raw_rate = compute_event_rate(vel, obj_size, bg_density, false_pos_rate)
                for th_name, theta in THRESHOLDS.items():
                    rows.append({
                        'object_size_px': obj_size,
                        'velocity_px_s':  vel,
                        'background':     bg_name,
                        'threshold_name': th_name,
                        'theta':          theta,
                        'event_rate_raw': raw_rate,
                        **compute_dvs_power(raw_rate, theta),
                    })
    return pd.DataFrame(rows)

# --- Plots ---

def plot_power_vs_velocity(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
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
    sub = df[(df['threshold_name'] == 'med_threshold') &
             (df['background'] == 'low_texture') &
             (df['object_size_px'] == 50)].sort_values('velocity_px_s')
    fig, ax = plt.subplots(figsize=(8, 5))
    x = sub['velocity_px_s'].astype(str)
    ax.bar(x, sub['power_static_mW'], label='Static (mW)', color='steelblue')
    ax.bar(x, sub['power_dynamic_mW'], label='Dynamic (mW)', color='tomato',
           bottom=sub['power_static_mW'])
    ax.set(title='DVS Power Breakdown (med_threshold, low_texture)',
           xlabel='Velocity (px/s)', ylabel='Power (mW)')
    ax.legend(); ax.grid(True, axis='y')
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
    print(df[(df['threshold_name'] == 'med_threshold') &
             (df['background'] == 'low_texture')].to_string(index=False))

    plot_power_vs_velocity(df, OUT_DIR)
    plot_power_vs_background(df, OUT_DIR)
    plot_power_vs_threshold(df, OUT_DIR)
    plot_static_vs_dynamic(df, OUT_DIR)
    print('Done. Results in:', OUT_DIR)