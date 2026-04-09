import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# --- Import Ramaa's scene model (for false event rates and reference params) ---

SCENE_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', 'Remaas-work'
))
sys.path.insert(0, SCENE_MODEL_PATH)

from visualcomputing import false_pos_rate  # type: ignore

# --- DVS Circuit Parameters (prof's model, defaults match Lichtsteiner 128×128 @ 3.3V) ---

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

# False event fractions
DVS_FP_RATE  = 0.10   # false positives — fire but shouldn't (waste power)
DVS_FN_RATE  = 0.05   # false negatives — should fire but don't (save power)

# --- DVS Noise Model (from v2e paper: Hu et al. 2021) ---
# Threshold mismatch: each pixel has its own θ drawn from N(θ_nominal, σ_θ × θ_nominal)
NOISE_SIGMA_THETA = 0.03     # 3% std — manufacturing variation in comparator threshold
# Shot noise: random events from photon arrival statistics (Poisson process)
NOISE_SHOT_RATE_HZ = 1.0     # ~1 event/pixel/sec — background activity noise
# Leak events: slow parasitic charge buildup triggers spurious events
NOISE_LEAK_RATE_HZ = 0.1     # ~0.1 event/pixel/sec — sub-threshold leakage

# --- Contrast Threshold Sweep ---

BASELINE_THETA = 0.20
THRESHOLDS = {
    "low_threshold":  0.10,
    "med_threshold":  0.20,
    "high_threshold": 0.40,
}

# --- Video / Image-Sequence Input ---
# Set VIDEO_PATH to a video file OR a directory of sequential images (e.g. MOT17 img1/).
# For image directories, set VIDEO_FPS manually.
VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'Sergeys-work',
    'MOT17', 'train', 'MOT17-04-SDP', 'img1'
)
VIDEO_FPS  = 30      # only used when loading from image directory
MAX_FRAMES = 300     # limit frames to keep memory/runtime reasonable (None = all)

# --- Video Frame Loading ---

def load_video_frames(video_path: str, fps_override: float = None,
                      max_frames: int = None):
    """
    Load frames from a video file OR an image directory.
    Converts to grayscale float32, clips to [1.0, 255.0].

    Returns: (frames, fps, H, W)
    """
    path = os.path.abspath(video_path)

    if os.path.isdir(path):
        # Image sequence (e.g. MOT17 img1/ directory)
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        img_files = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(exts)
        ])
        if max_frames:
            img_files = img_files[:max_frames]
        fps = fps_override or VIDEO_FPS
        frames = []
        for f in img_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            frames.append(np.clip(img, 1.0, 255.0))
    else:
        # Video file
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames.append(np.clip(gray, 1.0, 255.0))
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()

    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(frames)}: {path}")

    H, W = frames[0].shape
    print(f"Loaded {len(frames)} frames at {fps:.1f} fps, resolution {W}×{H}")
    return frames, fps, H, W


# --- Per-pixel DVS Event Detection ---

def compute_video_event_rates(frames: list, fps: float, thetas: list) -> pd.DataFrame:
    """
    Frame-by-frame per-pixel Δln(I) ≥ θ firing detection on real video frames.

    For each consecutive frame pair, computes |ln(I_curr) - ln(I_prev)| per pixel
    and counts pixels that cross each threshold θ. This is the exact firing condition
    from Lichtsteiner 2008 (eq. 5): an event fires when Δln(I) ≥ θ.

    event_rate = fired_pixels × fps  (events/sec)

    Returns a DataFrame with one row per (frame transition, theta).
    """
    rows = []
    log_prev = np.log(frames[0].astype(np.float64))

    for i in range(1, len(frames)):
        log_curr = np.log(frames[i].astype(np.float64))
        d_ln_I   = np.abs(log_curr - log_prev)

        for theta in thetas:
            fired      = int(np.sum(d_ln_I >= theta))
            event_rate = round(fired * fps, 1)
            rows.append({
                'frame_idx':   i,
                'theta':       theta,
                'fired_pixels': fired,
                'event_rate':  event_rate,
            })

        log_prev = log_curr

    return pd.DataFrame(rows)


# --- DVS Noise Model: Event Frame Generation ---

def generate_theta_map(H: int, W: int, theta_nominal: float,
                       seed: int = 42) -> np.ndarray:
    """
    Per-pixel threshold map with manufacturing mismatch.
    Each pixel's threshold is drawn from N(θ_nominal, σ_θ × θ_nominal).
    Clipped to [θ_nominal/3, θ_nominal×3] to avoid degenerate values.
    """
    rng = np.random.RandomState(seed)
    sigma = NOISE_SIGMA_THETA * theta_nominal
    theta_map = rng.normal(theta_nominal, sigma, size=(H, W))
    return np.clip(theta_map, theta_nominal / 3, theta_nominal * 3)


def generate_event_frames(frames: list, fps: float, theta: float,
                          noisy: bool = False, seed: int = 0):
    """
    Generate per-pixel binary event frames from video.

    Clean mode (noisy=False):
      - Event fires where |Δln(I)| ≥ θ (uniform threshold)

    Noisy mode (noisy=True):
      - Threshold mismatch: per-pixel θ from Gaussian (v2e σ_θ=3%)
      - Shot noise: random Poisson firings at NOISE_SHOT_RATE_HZ
      - Leak events: random Poisson firings at NOISE_LEAK_RATE_HZ

    Returns: list of (H, W) bool arrays (one per frame transition),
             and list of fired pixel counts per frame.
    """
    H, W = frames[0].shape
    rng  = np.random.RandomState(seed)
    dt   = 1.0 / fps   # time between frames in seconds

    # Per-pixel threshold (noisy = mismatch, clean = uniform)
    if noisy:
        theta_map = generate_theta_map(H, W, theta, seed=seed)
    else:
        theta_map = np.full((H, W), theta)

    # Shot + leak probability per pixel per frame
    p_shot = NOISE_SHOT_RATE_HZ * dt   # ~0.033 at 30fps
    p_leak = NOISE_LEAK_RATE_HZ * dt   # ~0.003 at 30fps

    event_frames = []
    fired_counts = []

    log_prev = np.log(frames[0].astype(np.float64))

    for i in range(1, len(frames)):
        log_curr = np.log(frames[i].astype(np.float64))
        d_ln_I   = np.abs(log_curr - log_prev)

        # Signal events: per-pixel threshold crossing
        events = d_ln_I >= theta_map

        if noisy:
            # Shot noise: random firings (Poisson)
            shot_noise = rng.random((H, W)) < p_shot
            # Leak events: slow parasitic firings
            leak_noise = rng.random((H, W)) < p_leak
            # Combine: signal OR shot OR leak
            events = events | shot_noise | leak_noise

        event_frames.append(events)
        fired_counts.append(int(np.sum(events)))
        log_prev = log_curr

    return event_frames, fired_counts


# --- Circuit-level Power Helpers (unchanged from animation model) ---

def compute_energy_per_event(theta: float) -> float:
    """Energy per event in nJ from circuit parameters."""
    e_afe    = (theta ** 2) * DVS_C2 * (1.0 + DVS_C2 / DVS_C1)
    c_sw_dig = ((DVS_CGATE + DVS_C3)
                + ((math.log2(640) + math.log2(480)) * DVS_CARB)
                + (0.5 * DVS_NUM_BITS * DVS_CBUS))
    e_dig    = (DVS_VDD ** 2) * c_sw_dig
    return (e_afe + e_dig) * 1e9   # J → nJ


def compute_static_power_mw(n_pixels: int) -> float:
    """
    Static power in mW for a sensor with n_pixels pixels.
    Uses actual frame dimensions rather than a fixed reference.
    """
    i_static = n_pixels * (DVS_IAFE + DVS_ILEAK) + DVS_IOTHER
    return DVS_VDD * i_static * 1e3   # W → mW


def compute_dvs_power(event_rate: float, theta: float, n_pixels: int) -> dict:
    """
    Compute DVS power from event rate, threshold, and pixel count.
    Same model as animation version — refractory cap, FP/FN breakdown.
    """
    e_per_event_nJ = compute_energy_per_event(theta)
    p_static_mW    = compute_static_power_mw(n_pixels)

    max_rate_array    = n_pixels / DVS_TREF
    event_rate_capped = min(event_rate, max_rate_array)

    r_fp    = DVS_FP_RATE * event_rate_capped
    r_fn    = DVS_FN_RATE * event_rate_capped
    r_valid = event_rate_capped - r_fn

    e_J            = e_per_event_nJ * 1e-9
    p_dyn_valid_mW = r_valid * e_J * 1e3
    p_dyn_fp_mW    = r_fp    * e_J * 1e3
    p_dyn_fn_mW    = r_fn    * e_J * 1e3
    p_total_mW     = p_static_mW + p_dyn_valid_mW + p_dyn_fp_mW

    return {
        'event_rate_eff':      round(event_rate_capped, 1),
        'e_per_event_nJ':      round(e_per_event_nJ, 4),
        'power_static_mW':     round(p_static_mW, 3),
        'power_dyn_valid_mW':  round(p_dyn_valid_mW, 3),
        'power_dyn_fp_mW':     round(p_dyn_fp_mW, 3),
        'power_dyn_fn_mW':     round(p_dyn_fn_mW, 3),
        'power_total_mW':      round(p_total_mW, 3),
    }


# --- Main Analysis ---

def run_video_analysis(video_path: str) -> tuple:
    """
    Load a video and compute DVS power frame-by-frame for all thresholds,
    both clean (ideal) and noisy (with v2e noise model).

    Returns: (df, fps, H, W, sample_event_frames)
      df  — DataFrame with per-frame power for clean + noisy
      fps — video frame rate
      H,W — frame dimensions
      sample_event_frames — dict of sample frames for visualization
    """
    frames, fps, H, W = load_video_frames(video_path, max_frames=MAX_FRAMES)
    n_pixels   = H * W
    th_name_map = {v: k for k, v in THRESHOLDS.items()}

    rows = []
    sample_event_frames = {}   # for visualization

    for th_name, theta in THRESHOLDS.items():
        print(f"  processing {th_name} (θ={theta}) ...", flush=True)

        # Clean (ideal sensor — uniform threshold, no noise)
        _, clean_counts = generate_event_frames(frames, fps, theta, noisy=False)
        # Noisy (realistic sensor — mismatch + shot + leak)
        noisy_ef, noisy_counts = generate_event_frames(frames, fps, theta, noisy=True)

        # Save a sample frame for visualization (frame 50 or last available)
        sample_idx = min(49, len(clean_counts) - 1)
        clean_ef_sample, _ = generate_event_frames(frames, fps, theta, noisy=False)
        sample_event_frames[th_name] = {
            'clean': clean_ef_sample[sample_idx],
            'noisy': noisy_ef[sample_idx],
        }

        for i in range(len(clean_counts)):
            frame_idx = i + 1
            for mode, counts in [('clean', clean_counts), ('noisy', noisy_counts)]:
                fired = counts[i]
                event_rate = round(fired * fps, 1)
                power_info = compute_dvs_power(event_rate, theta=theta, n_pixels=n_pixels)
                rows.append({
                    'frame_idx':      frame_idx,
                    'theta':          theta,
                    'threshold_name': th_name,
                    'mode':           mode,
                    'fired_pixels':   fired,
                    'event_rate':     event_rate,
                    **power_info,
                })

    return pd.DataFrame(rows), fps, H, W, sample_event_frames


# --- Plots ---

def plot_power_over_time(df: pd.DataFrame, fps: float, out_dir: str):
    """Power vs time — clean vs noisy for each threshold."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for ax, (th_name, th_df) in zip(axes, df.groupby('threshold_name')):
        for mode, mode_df in th_df.groupby('mode'):
            mode_df = mode_df.sort_values('frame_idx')
            time_s  = mode_df['frame_idx'] / fps
            color = '#cc4444' if mode == 'noisy' else '#2255AA'
            ax.plot(time_s, mode_df['power_total_mW'], linewidth=1.2,
                    linestyle='-', color=color, label=f'{mode}')

        p_static = th_df['power_static_mW'].iloc[0]
        ax.axhline(y=p_static, color='steelblue', linewidth=1, linestyle=':')
        ax.set(title=f'{th_name} (θ={th_df["theta"].iloc[0]:.2f})', ylabel='Power (mW)')
        ax.legend(loc='upper right'); ax.grid(True)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('DVS Power Over Time: Clean vs Noisy (real video)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_rv_power_over_time.png'), dpi=200)
    plt.close()


def plot_fired_pixels_over_time(df: pd.DataFrame, fps: float, out_dir: str):
    """Fired pixel count — clean vs noisy for each threshold."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for ax, (th_name, th_df) in zip(axes, df.groupby('threshold_name')):
        for mode, mode_df in th_df.groupby('mode'):
            mode_df = mode_df.sort_values('frame_idx')
            time_s  = mode_df['frame_idx'] / fps
            color = '#cc4444' if mode == 'noisy' else '#2255AA'
            ax.plot(time_s, mode_df['fired_pixels'], linewidth=1.0,
                    linestyle='-', color=color, label=f'{mode}')

        ax.set(title=f'{th_name} (θ={th_df["theta"].iloc[0]:.2f})', ylabel='Fired pixels')
        ax.legend(loc='upper right'); ax.grid(True)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('DVS Fired Pixels Per Frame: Clean vs Noisy\nΔln(I) ≥ θ per pixel',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_rv_fired_pixels.png'), dpi=200)
    plt.close()


def plot_avg_power_vs_threshold(df: pd.DataFrame, out_dir: str):
    """Average power vs threshold — clean vs noisy side by side."""
    summary = (df.groupby(['threshold_name', 'theta', 'mode'])['power_total_mW']
                 .mean().reset_index()
                 .sort_values('theta'))

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ['clean', 'noisy']:
        sub = summary[summary['mode'] == mode]
        color = '#cc4444' if mode == 'noisy' else '#2255AA'
        ax.plot(sub['theta'], sub['power_total_mW'], marker='o', linewidth=2,
                linestyle='-', color=color, label=mode)
        for _, r in sub.iterrows():
            ax.annotate(f"{r['power_total_mW']:.3f}",
                        (r['theta'], r['power_total_mW']),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=7, ha='center')

    p_static = df['power_static_mW'].iloc[0]
    ax.axhline(y=p_static, color='steelblue', linewidth=1.2, linestyle=':',
               label=f'Static floor ({p_static:.2f} mW)')

    ax.set(title='DVS Avg Power vs Threshold: Clean vs Noisy (real video)',
           xlabel='Threshold θ', ylabel='Avg DVS Total Power (mW)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_rv_avg_power_vs_threshold.png'), dpi=200)
    plt.close()


def plot_noise_impact_summary(df: pd.DataFrame, out_dir: str):
    """Fired pixels clean vs noisy per threshold, with power overhead as annotation."""
    summary = (df.groupby(['threshold_name', 'theta', 'mode'])
                 .agg(avg_power=('power_total_mW', 'mean'),
                      avg_fired=('fired_pixels', 'mean'))
                 .reset_index()
                 .sort_values('theta'))

    thetas = sorted(summary['theta'].unique())
    th_names = [summary[summary['theta'] == t]['threshold_name'].iloc[0] for t in thetas]

    clean_power = [summary[(summary['theta'] == t) & (summary['mode'] == 'clean')]['avg_power'].iloc[0] for t in thetas]
    noisy_power = [summary[(summary['theta'] == t) & (summary['mode'] == 'noisy')]['avg_power'].iloc[0] for t in thetas]
    clean_fired = [summary[(summary['theta'] == t) & (summary['mode'] == 'clean')]['avg_fired'].iloc[0] for t in thetas]
    noisy_fired = [summary[(summary['theta'] == t) & (summary['mode'] == 'noisy')]['avg_fired'].iloc[0] for t in thetas]

    fig, axes = plt.subplots(len(thetas), 1, figsize=(8, 4 * len(thetas)))

    for i, ax in enumerate(axes):
        x = np.arange(2)
        bars = ax.bar(x, [clean_fired[i], noisy_fired[i]], color=['#2255AA', '#cc4444'],
                       width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(['Clean', 'Noisy'])
        ax.set_ylabel('Avg fired pixels/frame')
        ax.grid(True, axis='y')

        # Percentage overhead
        overhead_pct = ((noisy_fired[i] - clean_fired[i]) / clean_fired[i] * 100
                        if clean_fired[i] > 0 else 0)
        ax.annotate(f'+{overhead_pct:.1f}% more pixels',
                    xy=(1, noisy_fired[i]), textcoords='offset points',
                    xytext=(0, 8), fontsize=9, ha='center', fontweight='bold')

        # Power overhead as text box
        power_oh = noisy_power[i] - clean_power[i]
        ax.text(0.98, 0.95,
                f'Power overhead: +{power_oh:.3f} mW\n'
                f'Clean: {clean_power[i]:.2f} mW | Noisy: {noisy_power[i]:.2f} mW',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc'))

        ax.set_title(f'{th_names[i]} (θ={thetas[i]:.2f})', fontweight='bold')

    plt.suptitle('DVS Noise Impact: Extra Fired Pixels Per Threshold\n'
                 'Noise sources: threshold mismatch (3%), shot noise (1 Hz/px), leak (0.1 Hz/px)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_rv_noise_impact.png'), dpi=200)
    plt.close()


def plot_signal_purity_tradeoff(df: pd.DataFrame, out_dir: str):
    """Stacked bar: signal vs noise pixels per threshold, with power on secondary axis."""
    summary = (df.groupby(['threshold_name', 'theta', 'mode'])
                 .agg(avg_power=('power_total_mW', 'mean'),
                      avg_fired=('fired_pixels', 'mean'))
                 .reset_index()
                 .sort_values('theta'))

    thetas = sorted(summary['theta'].unique())
    th_labels = [f"θ={t:.2f}" for t in thetas]

    signal = []    # real events (clean fired count)
    noise = []     # extra pixels from noise (noisy - clean)
    power_clean = []
    power_noisy = []

    for t in thetas:
        c = summary[(summary['theta'] == t) & (summary['mode'] == 'clean')].iloc[0]
        n = summary[(summary['theta'] == t) & (summary['mode'] == 'noisy')].iloc[0]
        signal.append(c['avg_fired'])
        noise.append(n['avg_fired'] - c['avg_fired'])
        power_clean.append(c['avg_power'])
        power_noisy.append(n['avg_power'])

    signal = np.array(signal)
    noise = np.array(noise)
    total = signal + noise
    purity = signal / total * 100

    fig, ax1 = plt.subplots(figsize=(9, 6))

    x = np.arange(len(thetas))
    w = 0.5
    ax1.bar(x, signal, w, label='Signal (real events)', color='#5588CC')
    ax1.bar(x, noise, w, bottom=signal, label='Noise (shot + leak + mismatch)', color='#dd7777')

    # Purity label inside the red (noise) area, just above the blue/red boundary
    for i in range(len(thetas)):
        label_y = signal[i] + noise[i] * 0.05  # just above the blue-red boundary
        ax1.text(x[i], label_y,
                 f'{purity[i]:.1f}% signal',
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color='black')

    ax1.set_xticks(x); ax1.set_xticklabels(th_labels)
    ax1.set_ylabel('Avg fired pixels/frame')
    ax1.legend(loc='upper left')
    ax1.grid(True, axis='y', alpha=0.3)

    # Power on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, power_clean, 'o-', color='#2255AA', label='Power (clean)', markersize=8)
    ax2.plot(x, power_noisy, 's-', color='#cc4444', label='Power (noisy)', markersize=8)
    ax2.set_ylabel('Avg power (mW)')
    ax2.legend(loc='upper right')

    plt.title('DVS Tradeoff: Signal Purity vs Power\n'
              'Higher θ saves power but noise dominates the output',
              fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_rv_signal_purity_tradeoff.png'), dpi=200)
    plt.close()


def plot_event_frame_comparison(sample_frames: dict, out_dir: str):
    """Side-by-side clean vs noisy event frames for each threshold."""
    n = len(sample_frames)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, (th_name, frames_dict) in enumerate(sorted(sample_frames.items())):
        for col, (mode, ef) in enumerate([('clean', frames_dict['clean']),
                                           ('noisy', frames_dict['noisy'])]):
            ax = axes[row, col]
            ax.imshow(ef.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
            fired = int(np.sum(ef))
            ax.set_title(f'{th_name} — {mode}\n({fired:,} fired pixels)', fontsize=10)
            ax.axis('off')

    plt.suptitle('DVS Event Frames: Clean vs Noisy (sample frame)\n'
                 'White = pixel fired an event', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dvs_rv_event_frames.png'), dpi=200)
    plt.close()


# --- Main ---

if __name__ == '__main__':
    if VIDEO_PATH is None:
        raise ValueError("Set VIDEO_PATH at the top of this file before running.")

    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dvs_results_real_video')
    os.makedirs(OUT_DIR, exist_ok=True)

    df, fps, H, W, sample_frames = run_video_analysis(VIDEO_PATH)
    df.to_csv(os.path.join(OUT_DIR, 'dvs_rv_summary.csv'), index=False)

    n_pixels = H * W
    print(f"P_static = {compute_static_power_mw(n_pixels):.2f} mW | "
          f"E_per_event (θ=0.20) = {compute_energy_per_event(BASELINE_THETA):.4f} nJ")

    plot_power_over_time(df, fps, OUT_DIR)
    plot_fired_pixels_over_time(df, fps, OUT_DIR)
    plot_avg_power_vs_threshold(df, OUT_DIR)
    plot_noise_impact_summary(df, OUT_DIR)
    plot_signal_purity_tradeoff(df, OUT_DIR)
    plot_event_frame_comparison(sample_frames, OUT_DIR)

    print('Done. Results in:', OUT_DIR)
