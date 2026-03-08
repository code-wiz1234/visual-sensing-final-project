# -*- coding: utf-8 -*-
"""
VisualComputing.ipynb → visualcomputing.py
Scene model used by ModuCIS runner.

Exports used by CIS runner:
  - object_sizes (px)
  - velocities (px/s)
  - safety_factor (scalar)
  - compute_fps_min(obj_speed_px_s, obj_size_px, safety)

Optional exports (for ADC/SNR path later):
  - compute_min_snr(obj_contrast, noise_floor=1.0)
  - backgrounds (edge density), false_pos_rate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Scene model parameters (px / px/s)
# -----------------------------
object_sizes = [25, 50, 100]               # small, medium, large object (pixels)
velocities   = [10, 50, 100, 200, 500]     # low to fast (pixels/second)
safety_factor = 10                         # frames per object-width crossing

# DVS-related (kept for comparison stage; not used by CIS runner directly)
false_pos_rate = 0.10                      # 10% of DVS events are noise/false triggers
backgrounds = {
    "low_texture": 0.05,                   # plain background (edge density fraction)
    "high_texture": 0.40                   # cluttered background
}

# Fixed scene settings (used for DVS plotting and documentation)
scene_width  = 640     # pixels
scene_height = 480     # pixels
light_wm2    = 5.0     # W/m² indoor lighting (for ModuCIS if needed)

# -----------------------------
# Analytical link: tracking → CIS/DVS requirements
# -----------------------------
def compute_fps_min(obj_speed_px_s: float, obj_size_px: float, safety: float = safety_factor) -> float:
    """
    Minimum CIS frame rate to limit inter-frame motion (pixels),
    per the scene model rule: fps_min = (velocity / object_size) * safety_factor
    """
    fps = (obj_speed_px_s / obj_size_px) * safety
    return round(fps, 2)

def compute_event_rate(obj_speed_px_s: float, obj_size_px: float, bg_edge_density: float,
                       fp_rate: float = false_pos_rate) -> float:
    """
    Predicted DVS event rate from moving object + background activity.
    Simple proxy: object perimeter ~ 4 * object_size (px).
    """
    edge_length = 4 * obj_size_px
    obj_events  = edge_length * obj_speed_px_s
    bg_events   = bg_edge_density * scene_width * scene_height * 0.01 * obj_speed_px_s
    total = (obj_events + bg_events) * (1 + fp_rate)
    return round(total, 1)

def compute_min_snr(obj_contrast: float, noise_floor: float = 1.0) -> float:
    """Minimum SNR (dB) needed for CIS detection given contrast."""
    snr = 20 * np.log10(obj_contrast / noise_floor)
    return round(snr, 2)

# -----------------------------
# Optional: run as a script to visualize tables/curves
# -----------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("SCENE MODEL OUTPUT")
    print("=" * 65)

    # --- CIS table: min FPS per (size, velocity) ---
    cis_rows = []
    for obj_size in object_sizes:
        for speed in velocities:
            fps = compute_fps_min(speed, obj_size, safety_factor)
            cis_rows.append({
                "Object Size (px)": obj_size,
                "Velocity (px/s)": speed,
                "Min FPS Required": fps,
                "Note": "→ For ModuCIS"
            })
    cis_df = pd.DataFrame(cis_rows)
    print("\n CIS: Minimum Frame Rate Requirements")
    print(cis_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("DVS: Predicted Event Rates ")
    print("=" * 65)

    # --- DVS table: event rates for backgrounds ---
    dvs_rows = []
    for bg_name, bg_density in backgrounds.items():
        for obj_size in object_sizes:
            for speed in velocities:
                evt_rate = compute_event_rate(speed, obj_size, bg_density, false_pos_rate)
                dvs_rows.append({
                    "Background": bg_name,
                    "Object Size (px)": obj_size,
                    "Velocity (px/s)": speed,
                    "Event Rate (ev/s)": evt_rate,
                    "Note": "→ For DVS model"
                })
    dvs_df = pd.DataFrame(dvs_rows)
    print("\n DVS: Predicted Event Rates")
    print(dvs_df.to_string(index=False))

    # --- Plots: CIS min FPS vs velocity, DVS event rate vs velocity ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graph 1: Min FPS vs Velocity for each object size
    ax1 = axes[0]
    for obj_size in object_sizes:
        fps_vals = [compute_fps_min(v, obj_size, safety_factor) for v in velocities]
        ax1.plot(velocities, fps_vals, marker='o', label=f"Object {obj_size}px")
    ax1.set_xlabel("Object Velocity (pixels/second)")
    ax1.set_ylabel("Minimum FPS Required (CIS)")
    ax1.set_title("CIS: Min FPS vs Object Velocity")
    ax1.legend()
    ax1.grid(True)

    # Graph 2: DVS Event Rate vs Velocity for each background type (example object size = 50 px)
    ax2 = axes[1]
    for bg_name, bg_density in backgrounds.items():
        evt_vals = [compute_event_rate(v, 50, bg_density, false_pos_rate) for v in velocities]
        ax2.plot(velocities, evt_vals, marker='s', label=f"{bg_name}")
    ax2.set_xlabel("Object Velocity (pixels/second)")
    ax2.set_ylabel("DVS Event Rate (events/second)")
    ax2.set_title("DVS: Event Rate vs Object Velocity")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Scene Model Outputs", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("scene_model_outputs.png", dpi=150)
    plt.show()

    print("\n Scene model complete!")
