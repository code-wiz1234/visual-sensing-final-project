"""Generate slide-ready PNGs from crossover and grid data."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MOT_ROOT = os.path.join(THIS_DIR, "MOT17")

# default output -- overridden by generate_all_slides() for synthetic
SLIDES_DIR = os.path.join(THIS_DIR, "real_world_slides")
os.makedirs(SLIDES_DIR, exist_ok=True)

# label shown in suptitles -- "Real-World MOT17" or "Synthetic Scenes"
SOURCE_LABEL = "Real-World MOT17"

sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..", "Ishs-work")))

DVS_COLORS = {
    "DVS: Lichtsteiner 2008": "#1b9e77",
    "DVS: DAVIS346": "#d95f02",
    "DVS: Samsung Gen3.1": "#7570b3",
    "DVS: Prophesee IMX636": "#e7298a",
}
CIS_COLORS = {
    "CIS: OV7251 640x480 10b": "#2196F3",     # bright blue
    "CIS: IMX327 1080p 12b": "#FF5722",        # deep orange
    "CIS: AR0234 1200p 10b": "#4CAF50",         # green
    "CIS: IMX462 1080p 12b": "#9C27B0",         # purple
}
ALL_COLORS = {**DVS_COLORS, **CIS_COLORS}

# Map raw sensor names to display names
_DISPLAY_NAMES = {
    "Lichtsteiner 2008":    "DVS: Lichtsteiner 2008",
    "DAVIS346":             "DVS: DAVIS346",
    "Samsung DVS-Gen3.1":   "DVS: Samsung Gen3.1",
    "Prophesee IMX636":     "DVS: Prophesee IMX636",
    "OV7251 (OmniVision)":  "CIS: OV7251 640x480 10b",
    "IMX327 (Sony)":        "CIS: IMX327 1080p 12b",
    "AR0234 (ON Semi)":     "CIS: AR0234 1200p 10b",
    "IMX462 (Sony)":        "CIS: IMX462 1080p 12b",
}


def _relabel(df):
    """Replace raw sensor names with DVS:/CIS: prefixed display names."""
    df = df.copy()
    df["sensor_name"] = df["sensor_name"].map(lambda sensor_name: _DISPLAY_NAMES.get(sensor_name, sensor_name))
    return df



plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "figure.facecolor": "white",
    "axes.facecolor": "#fafafa", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})


def _save(fig, name):
    path = os.path.join(SLIDES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")



def run_crossover_data(mot_root, seq, max_frames, workers):
    """Run the full crossover grid and return the dataframe."""
    from run_crossover import _run_one
    from sensor_database import DVS_SENSORS, CIS_SENSORS

    velocity_scales = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    trackers = ["centroid", "iou", "sort"]

    sensors = []
    for dvs_sensor in DVS_SENSORS:
        sensors.append({
            "name": dvs_sensor.name, "type": "DVS",
            "pixel_latency_s": dvs_sensor.pixel_latency_us * 1e-6,
            "refractory_cap": dvs_sensor.max_event_rate_mevps * 1e6,
            "resolution": dvs_sensor.resolution,
            "p_static_mw": dvs_sensor.p_static_mw,
            "e_per_event_nj": dvs_sensor.e_per_event_nj,
            "price": dvs_sensor.price_usd,
        })
    for cis_sensor in CIS_SENSORS:
        sensors.append({
            "name": cis_sensor.name, "type": "CIS",
            "max_fps": cis_sensor.max_fps,
            "resolution": cis_sensor.resolution,
            "adc_bits": cis_sensor.adc_bits,
            "power_idle_mw": cis_sensor.power_idle_mw,
            "power_max_mw": cis_sensor.power_at_max_fps_mw,
            "price": cis_sensor.price_usd,
        })

    configs = []
    # DVS sensors + CIS at max FPS (baseline)
    for sensor in sensors:
        for tracker_kind in trackers:
            for vel_scale in velocity_scales:
                configs.append({
                    "sensor": sensor, "tracker": tracker_kind, "velocity_scale": vel_scale,
                    "mot_root": mot_root, "seq": seq,
                    "max_frames": max_frames, "seed": 42,
                })

    # CIS FPS sweep: fixed 5fps and 15fps (below any source, always differentiates)
    cis_fps_settings = [5, 15]
    for cis_sensor in CIS_SENSORS:
        sensor_cfg = {
            "name": cis_sensor.name, "type": "CIS",
            "max_fps": cis_sensor.max_fps,
            "resolution": cis_sensor.resolution,
            "adc_bits": cis_sensor.adc_bits,
            "power_idle_mw": cis_sensor.power_idle_mw,
            "power_max_mw": cis_sensor.power_at_max_fps_mw,
            "price": cis_sensor.price_usd,
        }
        for fps_setting in cis_fps_settings:
            for tracker_kind in trackers:
                for vel_scale in velocity_scales:
                    configs.append({
                        "sensor": sensor_cfg, "tracker": tracker_kind,
                        "velocity_scale": vel_scale, "actual_fps": fps_setting,
                        "mot_root": mot_root, "seq": seq,
                        "max_frames": max_frames, "seed": 42,
                    })

    # DVS contrast threshold sweep: 1% (ultra-sensitive) to 10% (baseline)
    dvs_thresholds = [0.01, 0.03]
    for dvs_sensor in DVS_SENSORS:
        sensor_cfg = {
            "name": dvs_sensor.name, "type": "DVS",
            "pixel_latency_s": dvs_sensor.pixel_latency_us * 1e-6,
            "refractory_cap": dvs_sensor.max_event_rate_mevps * 1e6,
            "resolution": dvs_sensor.resolution,
            "p_static_mw": dvs_sensor.p_static_mw,
            "e_per_event_nj": dvs_sensor.e_per_event_nj,
            "price": dvs_sensor.price_usd,
        }
        for ct in dvs_thresholds:
            for tracker_kind in trackers:
                for vel_scale in velocity_scales:
                    configs.append({
                        "sensor": sensor_cfg, "tracker": tracker_kind,
                        "velocity_scale": vel_scale, "contrast_threshold": ct,
                        "mot_root": mot_root, "seq": seq,
                        "max_frames": max_frames, "seed": 42,
                    })

    print(f"Running {len(configs)} crossover configs on {workers} workers...")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, cfg): cfg for cfg in configs}
        for done_count, future in enumerate(as_completed(futures)):
            try:
                rows.append(future.result())
            except Exception as e:
                cfg = futures[future]
                print(f"  FAILED: {cfg['sensor']['name']} {cfg['tracker']} vs={cfg['velocity_scale']}: {e}")
            if (done_count + 1) % 40 == 0:
                print(f"  {done_count+1}/{len(configs)} done...")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(THIS_DIR, "crossover_results.csv"), index=False)
    return df



def _draw_power_bands(ax, df_all, sensor_type, tracker="sort"):
    """Draw power bands for one sensor type on a power axis."""
    sub = df_all[(df_all["sensor_type"] == sensor_type) & (df_all["tracker"] == tracker)]
    if sub.empty:
        return
    marker = "^" if sensor_type == "DVS" else "s"
    for name, grp in sub.groupby("sensor_name"):
        envelope = grp.groupby("mean_velocity_px_s")["power_mW"].agg(["min", "max"])
        envelope = envelope.sort_index()
        c = ALL_COLORS.get(name, "gray")
        has_range = (envelope["max"] - envelope["min"]).abs().max() > 0.5
        if has_range:
            ax.fill_between(envelope.index, envelope["min"], envelope["max"],
                            color=c, alpha=0.06)
            ax.plot(envelope.index, envelope["min"], color=c,
                    linestyle=":", linewidth=1, alpha=0.3)
        style = "--" if sensor_type == "CIS" else "-"
        ax.plot(envelope.index, envelope["max"], marker=marker, color=c,
                linestyle=style, linewidth=2, label=name)


def plot_01_power_vs_velocity(df):
    """DVS and CIS power with bands — both show operating range."""
    sort_df = df[df["tracker"] == "sort"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    _draw_power_bands(ax1, sort_df, "DVS")
    ax1.set(xlabel="Object Velocity (px/s)", ylabel="Power (mW)",
            title="DVS Power (band = θ 1%→10%)", xscale="log")
    ax1.legend(fontsize=7, bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)

    _draw_power_bands(ax2, sort_df, "CIS")
    ax2.set(xlabel="Object Velocity (px/s)", ylabel="Power (mW)",
            title="CIS Power (band = FPS 5→max)", xscale="log")
    ax2.legend(fontsize=7, bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)

    ymax = sort_df["power_mW"].max() * 1.15
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)

    fig.suptitle("Sensor Power Operating Range\n"
                 "DVS: θ knob (1-10% contrast threshold)  |  "
                 "CIS: FPS knob (5/15/max fps)",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "01_power_vs_velocity.png")


def _baseline_only(df):
    """Filter to baseline configs (θ=0.10 for DVS, max FPS for CIS)."""
    dvs = df[df["sensor_type"] == "DVS"]
    cis = df[df["sensor_type"] == "CIS"]
    if "contrast_threshold" in dvs.columns:
        dvs = dvs[(dvs["contrast_threshold"].isna()) | (dvs["contrast_threshold"] <= 0.101)]
    if "actual_fps" in cis.columns:
        cis = cis.sort_values("actual_fps", ascending=False).drop_duplicates(
            subset=["sensor_name", "velocity_scale", "tracker"], keep="first")
    return pd.concat([dvs, cis], ignore_index=True)


CIS_HATCHES = {
    "CIS: OV7251 640x480 10b": "",
    "CIS: IMX327 1080p 12b": "",
    "CIS: AR0234 1200p 10b": "",
    "CIS: IMX462 1080p 12b": "",
}
CIS_DASHES = {
    "CIS: OV7251 640x480 10b": (5, 2),
    "CIS: IMX327 1080p 12b": (8, 3),
    "CIS: AR0234 1200p 10b": (3, 2, 1, 2),
    "CIS: IMX462 1080p 12b": (1, 1),
}


def _draw_sensor_bands(ax, df_all, tracker):
    """Draw bands for both CIS (FPS range) and DVS (event-rate range)."""
    for stype, marker, hatches in [("CIS", "s", CIS_HATCHES), ("DVS", "^", {})]:
        sub = df_all[(df_all["sensor_type"] == stype) & (df_all["tracker"] == tracker)]
        if sub.empty:
            continue
        for name, grp in sub.groupby("sensor_name"):
            envelope = grp.groupby("mean_velocity_px_s")["mota"].agg(["min", "max"])
            envelope = envelope.sort_index()
            c = ALL_COLORS.get(name, "gray")
            hatch = hatches.get(name, "")
            has_range = (envelope["max"] - envelope["min"]).abs().max() > 0.005
            if has_range:
                ax.fill_between(envelope.index, envelope["min"], envelope["max"],
                                color=c, alpha=0.05)
                ax.plot(envelope.index, envelope["min"], linestyle=":",
                        color=c, alpha=0.4, linewidth=1)
            style = "--" if stype == "CIS" else "-"
            ax.plot(envelope.index, envelope["max"], linestyle=style,
                    color=c, marker=marker, markersize=4, linewidth=2,
                    label=name)


def plot_02_crossover_mota(df):
    """MOTA vs velocity, one panel per tracker. CIS as banded FPS ranges."""
    baseline = _baseline_only(df)
    trackers = ["centroid", "iou", "sort"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, tk in zip(axes, trackers):
        _draw_sensor_bands(ax, df, tk)
        ax.set(xlabel="Velocity (px/s)", title=f"{tk.upper()} Tracker", xscale="log")
        ax.legend(fontsize=5, bbox_to_anchor=(0.5, -0.18), loc="upper center", ncol=4)
    axes[0].set_ylabel("MOTA")

    fig.suptitle("Tracking Quality vs Object Velocity — All Trackers\n"
                 "(solid+band = DVS θ 1-10%; dashed+band = CIS FPS 5/15/max)",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_crossover_mota.png")


def plot_03_crossover_power(df):
    """Power vs velocity for all sensors with operating range bands."""
    sort_df = df[df["tracker"] == "sort"]
    fig, ax = plt.subplots(figsize=(10, 6))

    # Both sensor types with bands
    _draw_power_bands(ax, sort_df, "CIS")
    _draw_power_bands(ax, sort_df, "DVS")

    ax.set(xlabel="Object Velocity (px/s)", ylabel="Power (mW)",
           title="All Sensors: Power vs Velocity\n"
                 "(bands = operating range: DVS θ 1-10%, CIS FPS 5/15/max)",
           xscale="log", yscale="log")
    ax.legend(fontsize=6, bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=4)

    plt.tight_layout()
    _save(fig, "03_crossover_power.png")


def plot_04_mota_vs_power_scatter(df):
    """Scatter: MOTA vs power. CIS as FPS arcs, DVS as event-rate arcs."""
    sort_df = df[df["tracker"] == "sort"]
    # pick 3 representative velocity scales from whatever's in the data
    available = sorted(sort_df["velocity_scale"].unique())
    if len(available) >= 3:
        scales = [available[0], available[len(available) // 2], available[-2]]
    else:
        scales = available[:3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, vs in zip(axes, scales):
        sub = sort_df[sort_df["velocity_scale"] == vs]

        # CIS FPS arcs
        for name, grp in sub[sub["sensor_type"] == "CIS"].groupby("sensor_name"):
            grp = grp.sort_values("power_mW")
            c = ALL_COLORS.get(name, "gray")
            ax.plot(grp["power_mW"], grp["mota"], "-", color=c, alpha=0.4, linewidth=1.5)
            ax.scatter(grp["power_mW"], grp["mota"], c=c, marker="s", s=60,
                       edgecolors="k", zorder=5, alpha=0.7)

        # DVS event-rate arcs
        for name, grp in sub[sub["sensor_type"] == "DVS"].groupby("sensor_name"):
            grp = grp.sort_values("power_mW")
            c = ALL_COLORS.get(name, "gray")
            if len(grp) > 1:
                ax.plot(grp["power_mW"], grp["mota"], "-", color=c, alpha=0.4, linewidth=1.5)
            ax.scatter(grp["power_mW"], grp["mota"], c=c, marker="^", s=100,
                       edgecolors="k", zorder=6)

        # Legend instead of per-point labels
        handles = []
        for name in sorted(sub["sensor_name"].unique()):
            c = ALL_COLORS.get(name, "gray")
            m = "^" if "DVS" in name else "s"
            handles.append(plt.Line2D([0], [0], marker=m, color=c, linestyle="",
                                      markersize=7, label=name))
        ax.legend(handles=handles, fontsize=5.5,
                  bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  ncol=4, framealpha=0.8)
        ax.set(xlabel="Power (mW)", title=f"Velocity {vs} px/s", xscale="log")
        ax.axhline(y=0.8, color="green", alpha=0.3, linestyle=":")
    axes[0].set_ylabel("MOTA")

    fig.suptitle("MOTA vs Power (arcs = operating range: CIS FPS 5/15/max, DVS θ 1-10%)\n"
                 "top-left = best",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_mota_vs_power_scatter.png")


def plot_05_sensor_comparison(df):
    """Bar chart at a mid-range velocity with operating-range whiskers."""
    available = sorted(df["velocity_scale"].unique())
    mid_vel = available[len(available) // 3]  # pick ~1/3 into the range
    all_mid = df[(df["tracker"] == "sort") & (df["velocity_scale"] == mid_vel)]
    baseline = _baseline_only(df)
    base_mid = baseline[(baseline["tracker"] == "sort") & (baseline["velocity_scale"] == mid_vel)]
    base_mid = base_mid.sort_values("mota", ascending=False).drop_duplicates(
        subset=["sensor_name"], keep="first").sort_values("mota", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    y_pos = range(len(base_mid))
    names = base_mid["sensor_name"].tolist()
    colors = [ALL_COLORS.get(sensor_name, "gray") for sensor_name in names]

    ax1.barh(y_pos, base_mid["mota"].values, color=colors, edgecolor="k", alpha=0.8)
    for bar_idx, name in enumerate(names):
        grp = all_mid[all_mid["sensor_name"] == name]
        if len(grp) > 1:
            lo, hi = grp["mota"].min(), grp["mota"].max()
            if hi - lo > 0.002:
                ax1.plot([lo, hi], [bar_idx, bar_idx], color="k", linewidth=2, solid_capstyle="butt")
                ax1.plot([lo], [bar_idx], "|", color="k", markersize=8)
                ax1.plot([hi], [bar_idx], "|", color="k", markersize=8)
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set(xlabel="MOTA",
            title=f"Tracking Quality at {mid_vel} px/s\n(whiskers = operating range)")

    ax2.barh(y_pos, base_mid["power_mW"].values, color=colors, edgecolor="k", alpha=0.8)
    for bar_idx, name in enumerate(names):
        grp = all_mid[all_mid["sensor_name"] == name]
        if len(grp) > 1:
            lo, hi = grp["power_mW"].min(), grp["power_mW"].max()
            if hi - lo > 0.5:
                ax2.plot([lo, hi], [bar_idx, bar_idx], color="k", linewidth=2, solid_capstyle="butt")
                ax2.plot([lo], [bar_idx], "|", color="k", markersize=8)
                ax2.plot([hi], [bar_idx], "|", color="k", markersize=8)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set(xlabel="Power (mW)",
            title=f"Power at {mid_vel} px/s\n(whiskers = operating range)")

    fig.suptitle(f"Sensor Comparison at {mid_vel} px/s (SORT tracker)\n"
                 "Whiskers: CIS = FPS 5/15/max | DVS = θ 1-10%",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_sensor_comparison_at_crossover.png")


def plot_06_design_rule(df):
    """Summary design-rule visual with CIS FPS bands."""
    baseline = _baseline_only(df)
    sort_df = baseline[baseline["tracker"] == "sort"]

    # Best DVS and best CIS (at max FPS) at each velocity
    best = []
    for vs in sort_df["velocity_scale"].unique():
        sub = sort_df[sort_df["velocity_scale"] == vs]
        dvs_best = sub[sub["sensor_type"] == "DVS"].sort_values("mota", ascending=False).iloc[0]
        cis_best = sub[sub["sensor_type"] == "CIS"].sort_values("mota", ascending=False).iloc[0]
        best.append({
            "velocity": dvs_best["mean_velocity_px_s"],
            "dvs_mota": dvs_best["mota"], "dvs_power": dvs_best["power_mW"],
            "cis_mota": cis_best["mota"], "cis_power": cis_best["power_mW"],
        })
    bdf = pd.DataFrame(best).sort_values("velocity")

    # CIS min MOTA (lowest FPS) for band
    cis_all_sort = df[(df["sensor_type"] == "CIS") & (df["tracker"] == "sort")]
    cis_min = []
    for vs in sort_df["velocity_scale"].unique():
        sub = cis_all_sort[cis_all_sort["velocity_scale"] == vs]
        if sub.empty:
            continue
        row = sort_df[sort_df["velocity_scale"] == vs]
        vel = row[row["sensor_type"] == "DVS"]["mean_velocity_px_s"].iloc[0]
        cis_min.append({"velocity": vel, "cis_mota_min": sub["mota"].min()})
    cmin = pd.DataFrame(cis_min).sort_values("velocity")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # CIS band: max FPS to min FPS
    if not cmin.empty:
        merged = bdf.merge(cmin, on="velocity")
        ax1.fill_between(merged["velocity"], merged["cis_mota_min"],
                         merged["cis_mota"], color="#FF5722", alpha=0.06,
                         label="CIS range (FPS 5/15/max)")

    # DVS band: best θ to worst θ
    dvs_all_sort = df[(df["sensor_type"] == "DVS") & (df["tracker"] == "sort")]
    dvs_band = []
    for vs in sort_df["velocity_scale"].unique():
        sub = dvs_all_sort[dvs_all_sort["velocity_scale"] == vs]
        if sub.empty:
            continue
        vel = bdf[bdf["velocity"] == bdf["velocity"].iloc[0]]["velocity"].iloc[0]
        try:
            vel = sort_df[(sort_df["velocity_scale"] == vs) &
                          (sort_df["sensor_type"] == "DVS")]["mean_velocity_px_s"].iloc[0]
        except IndexError:
            continue
        dvs_band.append({"velocity": vel,
                         "dvs_mota_min": sub["mota"].min(),
                         "dvs_mota_max": sub["mota"].max()})
    if dvs_band:
        dband = pd.DataFrame(dvs_band).sort_values("velocity")
        ax1.fill_between(dband["velocity"], dband["dvs_mota_min"],
                         dband["dvs_mota_max"], color="#7570b3", alpha=0.05,
                         label="DVS range (θ 1-10%)")

    ax1.plot(bdf["velocity"], bdf["cis_mota"], "s--", color="#FF5722",
             label="Best CIS (100% FPS)", linewidth=2.5, markersize=10)
    ax1.plot(bdf["velocity"], bdf["dvs_mota"], "^-", color="#7570b3",
             label="Best DVS (θ=10%)", linewidth=2.5, markersize=10)
    ax1.fill_between(bdf["velocity"], bdf["dvs_mota"], bdf["cis_mota"],
                     where=bdf["dvs_mota"] >= bdf["cis_mota"],
                     alpha=0.07, color="#7570b3", label="DVS preferred zone")
    ax1.set(ylabel="MOTA", title="Tracking Quality: Best DVS vs Best CIS")
    ax1.set_xscale("log")
    ax1.legend(fontsize=7, bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=3)

    # Power
    ax2.plot(bdf["velocity"], bdf["dvs_power"], "^-", color="#7570b3",
             label="Best DVS", linewidth=2.5, markersize=10)
    ax2.plot(bdf["velocity"], bdf["cis_power"], "s--", color="#e6ab02",
             label="Best CIS (max FPS)", linewidth=2.5, markersize=10)
    ax2.fill_between(bdf["velocity"], bdf["dvs_power"], bdf["cis_power"],
                     alpha=0.06, color="gray", label="Power gap")
    ax2.set(xlabel="Object Velocity (px/s)", ylabel="Power (mW)",
            title="Power: Best DVS vs Best CIS")
    ax2.set_xscale("log")
    ax2.legend(fontsize=7, bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=3)

    # Find crossover via linear interpolation between bracketing points
    xover = None
    for vel_idx in range(len(bdf) - 1):
        cis_hi = bdf.iloc[vel_idx]["cis_mota"]
        cis_lo = bdf.iloc[vel_idx+1]["cis_mota"]
        dvs_hi = bdf.iloc[vel_idx]["dvs_mota"]
        dvs_lo = bdf.iloc[vel_idx+1]["dvs_mota"]
        if cis_hi > dvs_hi and dvs_lo >= cis_lo:
            # Linear interp: find t where cis(t) == dvs(t)
            # cis(t) = cis_hi + t*(cis_lo - cis_hi)
            # dvs(t) = dvs_hi + t*(dvs_lo - dvs_hi)
            denom = (cis_hi - dvs_hi) - (cis_lo - dvs_lo)
            if abs(denom) > 1e-9:
                t = (cis_hi - dvs_hi) / denom
                v_lo = bdf.iloc[vel_idx]["velocity"]
                v_hi = bdf.iloc[vel_idx+1]["velocity"]
                # Interpolate in log space (velocity is log-scaled)
                xover = v_lo * (v_hi / v_lo) ** t
            break
    if xover is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=xover, color="red", linestyle=":", linewidth=2)
        # Place label to the right of the line, not on top
        ax1.annotate(f"Crossover\n{xover:.0f} px/s",
                     xy=(xover, 0.92), xytext=(xover * 2.5, 0.95),
                     fontsize=11, color="red", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="red", linewidth=1.5),
                     ha="left", va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="red", alpha=0.9))

    fig.suptitle("Design Rule: DVS vs CIS Crossover\n"
                 "(SORT tracker | bands: CIS FPS 5/15/max, DVS θ 1-10%)",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "06_design_rule.png")


def plot_07_price_power_mota(df):
    """Price vs power vs MOTA. Legend-based labeling."""
    import re
    sort_df = df[(df["tracker"] == "sort") & (df["velocity_scale"] == 10)]

    def _price_mid(s):
        nums = re.findall(r'[\d,]+', s.replace(",", ""))
        if not nums:
            return None
        return sum(float(n) for n in nums) / len(nums)

    fig, ax = plt.subplots(figsize=(12, 8))
    handles = []

    # CIS: FPS trails (vertical lines at fixed price)
    for name, grp in sort_df[sort_df["sensor_type"] == "CIS"].groupby("sensor_name"):
        price = _price_mid(grp.iloc[0]["price"])
        if price is None:
            continue
        grp = grp.sort_values("power_mW")
        c = ALL_COLORS.get(name, "gray")
        best = grp.iloc[-1]
        for _, r in grp.iterrows():
            size = max(30, r["mota"] * 300)
            ax.scatter(price, r["power_mW"], c=c, marker="s", s=size,
                       edgecolors="k", zorder=5, alpha=0.6)
        ax.plot([price] * len(grp), grp["power_mW"], "-", color=c, alpha=0.4, linewidth=1.5)
        handles.append(plt.Line2D([0], [0], marker="s", color=c, linestyle="",
                                  markersize=8, label=f"{name} (MOTA={best['mota']:.2f})"))

    # DVS: show event-rate arcs if available
    for name, grp in sort_df[sort_df["sensor_type"] == "DVS"].groupby("sensor_name"):
        price = _price_mid(grp.iloc[0]["price"])
        if price is None:
            continue
        grp = grp.sort_values("power_mW")
        c = ALL_COLORS.get(name, "gray")
        best = grp.sort_values("mota", ascending=False).iloc[0]
        if len(grp) > 1:
            for _, r in grp.iterrows():
                size = max(30, r["mota"] * 300)
                ax.scatter(price, r["power_mW"], c=c, marker="^", s=size,
                           edgecolors="k", zorder=6, alpha=0.7)
            ax.plot([price] * len(grp), grp["power_mW"], "-", color=c, alpha=0.4, linewidth=1.5)
        else:
            ax.scatter(price, best["power_mW"], c=c, marker="^",
                       s=max(50, best["mota"] * 400),
                       edgecolors="k", zorder=6, alpha=0.9)
        handles.append(plt.Line2D([0], [0], marker="^", color=c, linestyle="",
                                  markersize=8, label=f"{name} (MOTA={best['mota']:.2f})"))

    ax.set(xlabel="Price (USD, approx)", ylabel="Power (mW)",
           title="Price vs Power vs Tracking at 10× Velocity\n"
                 "(size ~ MOTA; trails = CIS FPS 5/15/max / DVS θ 1-10%)",
           xscale="log", yscale="log")
    ax.legend(handles=handles, fontsize=7, bbox_to_anchor=(0.5, -0.12),
              loc="upper center", ncol=4, framealpha=0.9)
    plt.tight_layout()
    _save(fig, "07_price_power_mota.png")


def plot_08_operating_tradeoffs(df):
    """Side-by-side: CIS FPS% tradeoff vs DVS contrast-threshold% tradeoff.

    Left pair:  CIS AR0234 at 5/15/max FPS
    Right pair: DVS Samsung Gen3.1 at θ=1%/3%/10%
    """
    sort_df = df[df["tracker"] == "sort"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── Left column: CIS FPS tradeoff ──────────────────────────────────
    cis_name = _DISPLAY_NAMES.get("AR0234 (ON Semi)", "CIS: AR0234 1200p 10b")
    cis = sort_df[sort_df["sensor_name"] == cis_name]
    if cis.empty:
        cis_name = [sn for sn in sort_df["sensor_name"].unique() if "CIS" in sn][0]
        cis = sort_df[sort_df["sensor_name"] == cis_name]

    if not cis.empty and "fps_fraction" in cis.columns:
        fps_vals = sorted(cis["fps_fraction"].dropna().unique())
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(fps_vals)))
        for color, frac in zip(cmap, fps_vals):
            sub = cis[cis["fps_fraction"] == frac].sort_values("mean_velocity_px_s")
            if sub.empty:
                continue
            pwr = sub["power_mW"].iloc[0]
            label = f"FPS {int(frac*100)}% ({pwr:.0f} mW)"
            axes[0, 0].plot(sub["mean_velocity_px_s"], sub["mota"],
                            marker="s", color=color, label=label, linewidth=2)
            axes[1, 0].plot(sub["mean_velocity_px_s"], sub["power_mW"],
                            marker="s", color=color, label=label, linewidth=2)

    axes[0, 0].set(ylabel="MOTA", title=f"CIS: {cis_name}\nMOTA vs Velocity at various FPS%",
                   xscale="log")
    axes[0, 0].legend(fontsize=7, bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)
    axes[1, 0].set(xlabel="Velocity (px/s)", ylabel="Power (mW)",
                   title="Power vs Velocity", xscale="log")
    axes[1, 0].legend(fontsize=7, bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)

    # ── Right column: DVS contrast threshold tradeoff ──────────────────
    dvs_name = _DISPLAY_NAMES.get("Samsung DVS-Gen3.1", "DVS: Samsung Gen3.1")
    dvs = sort_df[sort_df["sensor_name"] == dvs_name]
    if dvs.empty:
        dvs_name = [sn for sn in sort_df["sensor_name"].unique() if "DVS" in sn][0]
        dvs = sort_df[sort_df["sensor_name"] == dvs_name]

    if not dvs.empty and "contrast_threshold" in dvs.columns:
        ct_vals = sorted(dvs["contrast_threshold"].dropna().unique())
        cmap = plt.cm.magma(np.linspace(0.2, 0.9, len(ct_vals)))
        for color, ct in zip(cmap, ct_vals):
            sub = dvs[dvs["contrast_threshold"] == ct].sort_values("mean_velocity_px_s")
            if sub.empty:
                continue
            pwr = sub["power_mW"].iloc[0]
            label = f"θ={int(ct*100)}% ({pwr:.0f} mW)"
            axes[0, 1].plot(sub["mean_velocity_px_s"], sub["mota"],
                            marker="^", color=color, label=label, linewidth=2)
            axes[1, 1].plot(sub["mean_velocity_px_s"], sub["power_mW"],
                            marker="^", color=color, label=label, linewidth=2)

    axes[0, 1].set(ylabel="MOTA", title=f"DVS: {dvs_name}\nMOTA vs Velocity at various θ%",
                   xscale="log")
    axes[0, 1].legend(fontsize=7, bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)
    axes[1, 1].set(xlabel="Velocity (px/s)", ylabel="Power (mW)",
                   title="Power vs Velocity", xscale="log")
    axes[1, 1].legend(fontsize=7, bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)

    fig.suptitle("Operating Knob Tradeoffs\n"
                 "CIS: lower FPS% → less power but more blur at high velocity\n"
                 "DVS: higher θ% → less power but sparser events at low velocity",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    _save(fig, "08_operating_tradeoffs.png")


# ── Synthetic baseline overlay slide ─────────────────────────────────

def plot_09_synthetic_vs_empirical(crossover_df):
    """Overlay synthetic baseline MOTA onto empirical crossover MOTA.

    Both use the same pipeline (sensor sim -> tracker -> MOTA), just
    different input: synthetic trajectories vs real MOT17 video.
    """
    SYNTHETIC_CSV = os.path.join(THIS_DIR, "..", "tracking_baseline_results.csv")
    MULTI_SENSOR_CSV = os.path.join(THIS_DIR, "..", "tracking_baseline_multi_sensor.csv")

    if not os.path.exists(SYNTHETIC_CSV):
        print("  [skip] 09_synthetic_vs_empirical: no tracking_baseline_results.csv")
        return

    synthetic = pd.read_csv(SYNTHETIC_CSV)
    multi_sensor = None
    if os.path.exists(MULTI_SENSOR_CSV):
        multi_sensor = pd.read_csv(MULTI_SENSOR_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # left: synthetic single-sensor MOTA
    ax = axes[0]
    sub = synthetic[(synthetic["object_size_px"] == 50) &
                    (synthetic["background"] == "low_texture")].sort_values("velocity_px_s")
    if not sub.empty:
        ax.plot(sub["velocity_px_s"], sub["dvs_mota"],
                "^-", color="#7570b3", linewidth=2.5, markersize=8, label="DVS (synthetic)")
        ax.plot(sub["velocity_px_s"], sub["cis_mota"],
                "s-", color="#FF5722", linewidth=2.5, markersize=8, label="CIS (synthetic)")
    ax.set(xlabel="Velocity (px/s)", ylabel="MOTA",
           title="Synthetic Baseline\n(50px, low texture, SORT)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # center: multi-sensor synthetic MOTA
    ax = axes[1]
    if multi_sensor is not None:
        for dvs_name in sorted(multi_sensor["dvs_sensor"].unique()):
            grp = multi_sensor[multi_sensor["dvs_sensor"] == dvs_name]
            avg = grp.groupby("velocity_px_s")["dvs_mota"].mean()
            display = _DISPLAY_NAMES.get(dvs_name, dvs_name)
            ax.plot(avg.index, avg.values, "^--", linewidth=1.5,
                    color=ALL_COLORS.get(display, "gray"), label=display)
        for cis_name in sorted(multi_sensor["cis_sensor"].unique()):
            grp = multi_sensor[multi_sensor["cis_sensor"] == cis_name]
            avg = grp.groupby("velocity_px_s")["cis_mota"].mean()
            display = _DISPLAY_NAMES.get(cis_name, cis_name)
            ax.plot(avg.index, avg.values, "s-", linewidth=1.5,
                    color=ALL_COLORS.get(display, "gray"), label=display)
        ax.set_title("Multi-Sensor Synthetic\n(all 8 sensors, 50px)")
    else:
        ax.text(0.5, 0.5, "Run tracking_baseline.py\nfor multi-sensor data",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_title("Multi-Sensor (not generated)")
    ax.set(xlabel="Velocity (px/s)", ylabel="MOTA")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # right: empirical MOTA from crossover
    ax = axes[2]
    sort_df = crossover_df[crossover_df["tracker"] == "sort"]
    baseline = _baseline_only(crossover_df)
    sort_base = baseline[baseline["tracker"] == "sort"]

    for sensor_name, grp in sort_base.groupby("sensor_name"):
        grp = grp.sort_values("mean_velocity_px_s")
        color = ALL_COLORS.get(sensor_name, "gray")
        marker = "^" if "DVS" in sensor_name else "s"
        style = "-" if "DVS" in sensor_name else "--"
        ax.plot(grp["mean_velocity_px_s"], grp["mota"],
                marker=marker, linestyle=style, color=color,
                linewidth=1.5, label=sensor_name)

    ax.set(xlabel="Velocity (px/s)", ylabel="MOTA",
           title="Empirical MOT17 Crossover\n(SORT tracker, real detections)")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Synthetic vs Empirical: Same Pipeline, Different Input\n"
                 "Left: synthetic trajectories | Center: all sensors | Right: real MOT17",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "09_synthetic_vs_empirical.png")


# ── Main ──────────────────────────────────────────────────────────────

def generate_all_slides(df, output_dir=None, source_label=None):
    """Generate the full set of slides from a crossover-format DataFrame.

    This is the shared entry point -- works for both real-world and synthetic data
    as long as the CSV has the same schema (sensor_name, sensor_type, tracker,
    velocity_scale, mean_velocity_px_s, power_mW, mota, idf1, ...).
    """
    global SLIDES_DIR, SOURCE_LABEL
    if output_dir is not None:
        SLIDES_DIR = output_dir
        os.makedirs(SLIDES_DIR, exist_ok=True)
    if source_label is not None:
        SOURCE_LABEL = source_label

    df = _relabel(df)

    print(f"\nGenerating slides in {SLIDES_DIR}/...")
    plot_01_power_vs_velocity(df)
    plot_02_crossover_mota(df)
    plot_03_crossover_power(df)
    plot_04_mota_vs_power_scatter(df)
    plot_05_sensor_comparison(df)
    plot_06_design_rule(df)
    plot_07_price_power_mota(df)
    plot_08_operating_tradeoffs(df)

    sort_df = df[df["tracker"] == "sort"]
    print(f"\nMOTA TABLE (SORT tracker):")
    pivot = sort_df.pivot_table(index="sensor_name", columns="velocity_scale",
                                values="mota", aggfunc="first")
    print(pivot.round(3).to_string())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-root", required=False, default=DEFAULT_MOT_ROOT)
    ap.add_argument("--seq", default="MOT17-04-SDP")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--reuse-data", action="store_true",
                    help="Skip recomputation, use existing crossover_results.csv")
    args = ap.parse_args()

    t0 = time.time()

    csv_path = os.path.join(THIS_DIR, "crossover_results.csv")
    if args.reuse_data and os.path.exists(csv_path):
        print(f"Reusing existing {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = run_crossover_data(args.mot_root, args.seq, args.max_frames, args.workers)

    real_world_dir = os.path.join(THIS_DIR, "..", "real_world_slides")
    generate_all_slides(df, output_dir=real_world_dir, source_label="Real-World MOT17")

    # also generate the overlay slide if synthetic data exists
    plot_09_synthetic_vs_empirical(df)

    print(f"\nAll done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
