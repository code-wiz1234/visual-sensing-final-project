# -*- coding: utf-8 -*-
"""
visualcomputing.py  —  Ramaa (Scene Model)
Team 4: CIS vs DVS Co-Design for Object Tracking

IMPORT SAFETY:
  All plotting and animation code is inside if __name__ == '__main__':
  so Harshitha (CIS) and Ishita (DVS) can safely import without
  any plots or animations auto-running.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# ══════════════════════════════════════════════════════════════════════════
#  SCENE PARAMETERS — imported by Harshitha and Ishita
# ══════════════════════════════════════════════════════════════════════════
object_sizes   = [25, 50, 100]
velocities     = [10, 50, 100, 200, 500]
velocities_viz = [50, 500, 2000]    # used in animation AND exported to Harshitha
safety_factor  = 10
false_pos_rate = 0.10

backgrounds = {
    "low_texture":  0.05,
    "high_texture": 0.40
}

scene_width  = 640
scene_height = 480
light_wm2    = 5.0    # indoor lighting — for Harshitha's ModuCIS pd_E

# ══════════════════════════════════════════════════════════════════════════
#  ANIMATION PARAMETERS — exported for Harshitha's CIS code
#  If you change your animation update these 3 values and
#  Harshitha's code automatically picks up the new values
# ══════════════════════════════════════════════════════════════════════════
ANIM_BG       = "low_texture"   # background used in animation
ANIM_OBJ_SIZE = 50              # object size used in animation

# ══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS — imported by Harshitha and Ishita
# ══════════════════════════════════════════════════════════════════════════
def compute_fps_min(obj_speed, obj_size, safety=10):
    fps = (obj_speed / obj_size) * safety
    return round(fps, 2)

def compute_event_rate(obj_speed, obj_size, bg_edge_density, fp_rate=0.10):
    edge_length = 4 * obj_size
    obj_events  = edge_length * obj_speed
    bg_events   = bg_edge_density * scene_width * scene_height * 0.01 * obj_speed
    total       = (obj_events + bg_events) * (1 + fp_rate)
    return round(total, 1)

def compute_min_snr(obj_contrast, noise_floor=1.0):
    snr = 20 * np.log10(obj_contrast / noise_floor)
    return round(snr, 2)

# ══════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS — imported by Harshitha
# ══════════════════════════════════════════════════════════════════════════
def draw_room(ax, bg_type):
    ax.add_patch(patches.Rectangle((0, 0), scene_width, 130,
                 facecolor="#8B7355", zorder=1))
    ax.add_patch(patches.Rectangle((0, 130), scene_width, 350,
                 facecolor="#D4C9B0", zorder=1))
    ax.add_patch(patches.Rectangle((0, 128), scene_width, 5,
                 facecolor="#A0906C", zorder=2))
    ax.add_patch(patches.Rectangle((470, 270), 120, 150,
                 facecolor="#87CEEB", zorder=2))
    ax.add_patch(patches.Rectangle((470, 270), 120, 150,
                 facecolor="none", edgecolor="#6B4A2A", linewidth=3, zorder=3))
    ax.add_patch(patches.Rectangle((528, 270), 4, 150,
                 facecolor="#6B4A2A", zorder=3))
    ax.add_patch(patches.Rectangle((470, 343), 120, 4,
                 facecolor="#6B4A2A", zorder=3))
    if bg_type == "high_texture":
        ax.add_patch(patches.Rectangle((18, 200), 85, 190,
                     facecolor="#7B5B2A", zorder=2))
        for i, c in enumerate(["#CC2200","#004488","#228B22","#AA6600","#880088"]):
            ax.add_patch(patches.Rectangle((23, 210 + i*34), 75, 26,
                         facecolor=c, zorder=3))
        ax.add_patch(patches.Rectangle((470, 162), 150, 12,
                     facecolor="#5C3A1A", zorder=2))
        for tx in [475, 608]:
            ax.add_patch(patches.Rectangle((tx, 130), 8, 34,
                         facecolor="#5C3A1A", zorder=2))
        ax.add_patch(patches.Rectangle((235, 295), 88, 68,
                     facecolor="#5577AA", zorder=2))
        ax.add_patch(patches.Rectangle((233, 293), 92, 72,
                     facecolor="none", edgecolor="#5C3D11", linewidth=3, zorder=3))
        np.random.seed(7)
        for _ in range(6):
            rx = np.random.uniform(130, 430)
            ry = np.random.uniform(15, 75)
            rw = np.random.uniform(25, 55)
            rh = np.random.uniform(12, 28)
            rc = np.random.choice(["#AA4400","#334455","#556633","#664422"])
            ax.add_patch(patches.Rectangle((rx, ry), rw, rh,
                         facecolor=rc, alpha=0.8, zorder=2))

def draw_sphere(ax, cx, cy, r, vel):
    ax.add_patch(patches.Ellipse((cx, 105), r*1.4, r*0.22,
                 facecolor="black", alpha=0.18, zorder=3))
    for i, offset in enumerate([-r*3.0, -r*1.6]):
        ax.add_patch(Circle((cx+offset, cy), r,
                     color="#4488DD", alpha=0.12+0.13*i, zorder=3))
    ax.add_patch(Circle((cx, cy), r, color="#2255AA", zorder=5))
    ax.add_patch(Circle((cx-r*0.33, cy+r*0.33), r*0.30,
                 color="white", alpha=0.5, zorder=6))
    ax.add_patch(Circle((cx+r*0.28, cy-r*0.28), r*0.11,
                 color="white", alpha=0.25, zorder=6))
    arrow_len = max(40, min(100, vel * 0.18))
    ax.annotate("", xy=(cx+r+arrow_len, cy), xytext=(cx+r+5, cy),
                arrowprops=dict(arrowstyle="->", color="yellow",
                                lw=2, mutation_scale=16), zorder=9)
    ax.text(cx+r+arrow_len+5, cy+6, f"{vel}px/s",
            color="yellow", fontsize=7, fontweight="bold", zorder=9)

# ══════════════════════════════════════════════════════════════════════════
#  ALL EXECUTION CODE — only runs when file is run directly
#  Harshitha and Ishita import safely above this block
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── ORIGINAL: Static scene image ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#111111")
    bg_titles = ["Simple Background (Low Texture)",
                 "Cluttered Background (High Texture)"]
    for ax, bg_key, title in zip(axes, list(backgrounds.keys()), bg_titles):
        draw_room(ax, bg_key)
        draw_sphere(ax, cx=320, cy=185, r=50, vel=100)
        ax.set_xlim(0, scene_width)
        ax.set_ylim(0, scene_height)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=12,
                     fontweight="bold", pad=10)
    handles = [
        patches.Patch(color="#2255AA", label="Sphere (object)"),
        patches.Patch(color="#4488DD", alpha=0.3, label="Motion trail"),
        plt.Line2D([0],[0], color="yellow", lw=2, label="Velocity arrow"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               facecolor="#2a2a3e", labelcolor="white",
               fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Scene Model — Sphere Tracking in a Room  |  Velocity: 100 px/s",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout(pad=2.0)
    plt.savefig("scene_model_graphic.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("Saved: scene_model_graphic.png")

    # ── ORIGINAL: Temporal variation ──────────────────────────────────────
    time_steps         = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    velocity_over_time = [10, 50, 100, 200, 500, 200, 100, 50, 10]
    obj_size = 50
    fps_over_time         = [compute_fps_min(v, obj_size, safety_factor)
                             for v in velocity_over_time]
    events_low_over_time  = [compute_event_rate(v, obj_size, 0.05, false_pos_rate)
                             for v in velocity_over_time]
    events_high_over_time = [compute_event_rate(v, obj_size, 0.40, false_pos_rate)
                             for v in velocity_over_time]

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes2[0]
    ax1.plot(time_steps, velocity_over_time, marker='o',
             color="#4488DD", linewidth=2.5, label="Object velocity")
    ax1.fill_between(time_steps, velocity_over_time, alpha=0.15, color="#4488DD")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Velocity (px/s)")
    ax1.set_title("Object Velocity Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    ax2 = axes2[1]
    ax2.plot(time_steps, events_low_over_time, marker='s',
             color="#22AA66", linewidth=2.5, label="DVS — low texture")
    ax2.plot(time_steps, events_high_over_time, marker='s',
             color="#FF6622", linewidth=2.5, label="DVS — high texture")
    cis_line = [max(fps_over_time) * 500] * len(time_steps)
    ax2.axhline(y=cis_line[0], color="#AAAAAA", linewidth=2,
                linestyle="--", label="CIS (constant power)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("DVS Event Rate (events/sec)")
    ax2.set_title("DVS Event Rate Over Time vs CIS")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    plt.suptitle("Temporal Variation — Scene Activity Changes Over Time",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("temporal_variation.png", dpi=150)
    plt.show()
    print("Saved: temporal_variation.png")

    # ── ORIGINAL: Tables ──────────────────────────────────────────────────
    print("\nTemporal Variation — Same object changing speed over time")
    print("=" * 70)
    temporal_rows = []
    for t, vel in zip(time_steps, velocity_over_time):
        fps    = compute_fps_min(vel, obj_size, safety_factor) if vel > 0 else 0
        e_low  = compute_event_rate(vel, obj_size, 0.05, false_pos_rate)
        e_high = compute_event_rate(vel, obj_size, 0.40, false_pos_rate)
        temporal_rows.append({
            "Time (s)":        t,
            "Velocity (px/s)": vel,
            "CIS Min FPS":     fps,
            "DVS Events Low":  e_low,
            "DVS Events High": e_high
        })
    print(pd.DataFrame(temporal_rows).to_string(index=False))

    print("=" * 65)
    print("SCENE MODEL OUTPUT")
    print("=" * 65)
    cis_rows = []
    for obj_size in object_sizes:
        for speed in velocities:
            fps = compute_fps_min(speed, obj_size, safety_factor)
            cis_rows.append({
                "Object Size (px)": obj_size,
                "Velocity (px/s)":  speed,
                "Min FPS Required": fps,
                "Note": "For ModuCIS"
            })
    print("\nCIS: Minimum Frame Rate Requirements")
    print(pd.DataFrame(cis_rows).to_string(index=False))

    print("\n" + "=" * 65)
    print("DVS: Predicted Event Rates")
    print("=" * 65)
    dvs_rows = []
    for bg_name, bg_density in backgrounds.items():
        for obj_size in object_sizes:
            for speed in velocities:
                evt_rate = compute_event_rate(speed, obj_size,
                                              bg_density, false_pos_rate)
                dvs_rows.append({
                    "Background":        bg_name,
                    "Object Size (px)":  obj_size,
                    "Velocity (px/s)":   speed,
                    "Event Rate (ev/s)": evt_rate,
                    "Note": "For DVS model"
                })
    print(pd.DataFrame(dvs_rows).to_string(index=False))

    # ── ORIGINAL: Scene model output plots ────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes3[0]
    for obj_size in object_sizes:
        fps_vals = [compute_fps_min(v, obj_size, safety_factor)
                    for v in velocities]
        ax1.plot(velocities, fps_vals, marker='o', label=f"Object {obj_size}px")
    ax1.set_xlabel("Object Velocity (pixels/second)")
    ax1.set_ylabel("Minimum FPS Required (CIS)")
    ax1.set_title("CIS: Min FPS vs Object Velocity")
    ax1.legend()
    ax1.grid(True)
    ax2 = axes3[1]
    for bg_name, bg_density in backgrounds.items():
        evt_vals = [compute_event_rate(v, 50, bg_density, false_pos_rate)
                    for v in velocities]
        ax2.plot(velocities, evt_vals, marker='s', label=bg_name)
    ax2.set_xlabel("Object Velocity (pixels/second)")
    ax2.set_ylabel("DVS Event Rate (events/second)")
    ax2.set_title("DVS: Event Rate vs Object Velocity")
    ax2.legend()
    ax2.grid(True)
    plt.suptitle("Scene Model Outputs", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("scene_model_outputs.png", dpi=150)
    plt.show()
    print("\nScene model complete!")

    # ══════════════════════════════════════════════════════════════════════
    #  NEW: Camera animation — sphere moves at 3 velocities
    #  Both backgrounds side by side
    #  Harshitha's code takes draw_room() and adds CIS snapshot view
    # ══════════════════════════════════════════════════════════════════════
    ANIM_R_     = 30
    ANIM_Y_     = 185
    ANIM_X0_    = 60
    ANIM_STEPS_ = 60

    def _anim_pos(vel, n=ANIM_STEPS_):
        ppf = max(4, min(scene_width * 0.8 / n, vel / velocities_viz[0] * 6))
        return [min(ANIM_X0_ + i * ppf, scene_width - ANIM_R_ - 10)
                for i in range(n)]

    # Build frame list across all 3 velocity phases
    anim_frames_ = []
    for vel in velocities_viz:
        fps_min_     = compute_fps_min(vel, ANIM_R_ * 2, safety_factor)
        xs_          = _anim_pos(vel)
        fps_scaled_  = max(1, min(ANIM_STEPS_,
                           int(fps_min_ / velocities_viz[0] * 3)))
        cis_interval_= max(1, ANIM_STEPS_ // fps_scaled_)
        for i, x in enumerate(xs_):
            anim_frames_.append((vel, x, fps_min_, i, cis_interval_))

    # Figure — both backgrounds side by side, camera view only
    fig_anim, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(16, 7))
    fig_anim.patch.set_facecolor("#111111")
    fig_anim.suptitle(
        "Scene Animation — Camera View\n"
        f"Sphere moving at velocities: {velocities_viz} px/s",
        color="white", fontsize=12, fontweight="bold"
    )

    for ax, bg_key, title in [
        (ax_low,  "low_texture",  "Simple Background (Low Texture)"),
        (ax_high, "high_texture", "Cluttered Background (High Texture)")
    ]:
        draw_room(ax, bg_key)
        ax.set_xlim(0, scene_width)
        ax.set_ylim(0, scene_height)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=11,
                     fontweight="bold", pad=8)

    # Sphere — low texture
    s_sph_l  = Circle((ANIM_X0_, ANIM_Y_), ANIM_R_, color="#2255AA", zorder=5)
    s_shi_l  = Circle((ANIM_X0_-ANIM_R_*.33, ANIM_Y_+ANIM_R_*.33),
                      ANIM_R_*.30, color="white", alpha=0.5, zorder=6)
    s_shd_l  = patches.Ellipse((ANIM_X0_, 105), ANIM_R_*1.4, ANIM_R_*.22,
                                facecolor="black", alpha=0.18, zorder=3)
    x_hist_l = [ANIM_X0_] * 3
    trails_l = [Circle((ANIM_X0_, ANIM_Y_), ANIM_R_,
                        color="#4488DD", alpha=0.12*(i+1), zorder=3)
                for i in range(3)]
    for p in [s_shd_l, s_sph_l, s_shi_l, *trails_l]:
        ax_low.add_patch(p)

    # Sphere — high texture
    s_sph_h  = Circle((ANIM_X0_, ANIM_Y_), ANIM_R_, color="#2255AA", zorder=5)
    s_shi_h  = Circle((ANIM_X0_-ANIM_R_*.33, ANIM_Y_+ANIM_R_*.33),
                      ANIM_R_*.30, color="white", alpha=0.5, zorder=6)
    s_shd_h  = patches.Ellipse((ANIM_X0_, 105), ANIM_R_*1.4, ANIM_R_*.22,
                                facecolor="black", alpha=0.18, zorder=3)
    x_hist_h = [ANIM_X0_] * 3
    trails_h = [Circle((ANIM_X0_, ANIM_Y_), ANIM_R_,
                        color="#4488DD", alpha=0.12*(i+1), zorder=3)
                for i in range(3)]
    for p in [s_shd_h, s_sph_h, s_shi_h, *trails_h]:
        ax_high.add_patch(p)

    # Text overlays
    t_low  = ax_low.text(10,  scene_height-18, '', color='yellow',
                         fontsize=9, fontweight='bold', zorder=10)
    t_high = ax_high.text(10, scene_height-18, '', color='yellow',
                          fontsize=9, fontweight='bold', zorder=10)

    def _update_cam(fi):
        vel, x, fps_min_, pi, ci = anim_frames_[fi]
        # Low texture
        x_hist_l.append(x); x_hist_l.pop(0)
        for tp, hx in zip(trails_l, x_hist_l): tp.center = (hx, ANIM_Y_)
        s_sph_l.center = (x, ANIM_Y_)
        s_shi_l.center = (x-ANIM_R_*.33, ANIM_Y_+ANIM_R_*.33)
        s_shd_l.set_center((x, 105))
        t_low.set_text(f"vel = {vel} px/s  |  min FPS = {fps_min_:.0f}")
        # High texture
        x_hist_h.append(x); x_hist_h.pop(0)
        for tp, hx in zip(trails_h, x_hist_h): tp.center = (hx, ANIM_Y_)
        s_sph_h.center = (x, ANIM_Y_)
        s_shi_h.center = (x-ANIM_R_*.33, ANIM_Y_+ANIM_R_*.33)
        s_shd_h.set_center((x, 105))
        t_high.set_text(f"vel = {vel} px/s  |  min FPS = {fps_min_:.0f}")
        return (*trails_l, s_sph_l, s_shi_l, s_shd_l,
                *trails_h, s_sph_h, s_shi_h, s_shd_h,
                t_low, t_high)

    cam_anim = FuncAnimation(fig_anim, _update_cam,
                             frames=len(anim_frames_),
                             interval=60, blit=True)

    # Save to same folder as this file
    save_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        from matplotlib.animation import FFMpegWriter
        out_path = os.path.join(save_dir, "scene_camera_animation.mp4")
        cam_anim.save(out_path,
                      writer=FFMpegWriter(fps=16, bitrate=2000),
                      savefig_kwargs={'facecolor': '#111111'})
        print(f"Saved: {out_path}")
    except Exception:
        out_path = os.path.join(save_dir, "scene_camera_animation.gif")
        cam_anim.save(out_path, writer='pillow', fps=16,
                      savefig_kwargs={'facecolor': '#111111'})
        print(f"Saved: {out_path}  (GIF — ffmpeg not found)")

    plt.show()
