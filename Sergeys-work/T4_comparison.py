"""
CIS vs DVS Power Comparison and Design Rules
Author: Sergey Petrushkevich
EECE5698 - Visual Sensing and Computing

Merges CIS (Harshita) and DVS (Ish) model outputs under the shared scene model (Ramaa)
to produce comparison plots and design rules for object tracking.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from numpy.polynomial.polynomial import polyfit

# figure out where everything lives relative to this script
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)

# Ish's DVS sweep results 90 rows covering all scene combos
DVS_CSV = os.path.join(
    PROJECT, "Ishs-work", "dvs_results", "dvs_all_scenes_summary.csv"
)

# Harshita's CIS model with background texture factored in 30 rows
CIS_BG_CSV = os.path.join(
    PROJECT,
    "Harshitas-work-S26-ModuCIS-modeling-main",
    "ModuCIS.-CIS-modeling-main",
    "ModuCIS.-CIS-modeling-main",
    "CIS_Model",
    "sweeps_results_final_cis_withbg",
    "cis_all_scenes_summary_with_bg.csv",
)

# resolution and fps sweeps from Harshita's CIS use-case analysis
RES_SWEEP_CSV = os.path.join(
    PROJECT,
    "Harshitas-work-S26-ModuCIS-modeling-main",
    "ModuCIS.-CIS-modeling-main",
    "ModuCIS.-CIS-modeling-main",
    "CIS_Model",
    "Use_cases",
    "sweeps_results_final_cis_model",
    "resolution_sweep",
    "res_sweep_detailed.csv",
)

FPS_SWEEP_CSV = os.path.join(
    PROJECT,
    "Harshitas-work-S26-ModuCIS-modeling-main",
    "ModuCIS.-CIS-modeling-main",
    "ModuCIS.-CIS-modeling-main",
    "CIS_Model",
    "Use_cases",
    "sweeps_results_final_cis_model",
    "fsp_sweep",
    "fps_sweep_detailed.csv",
)

OUT_DIR = BASE  # output directly into Sergeys-work/
os.makedirs(OUT_DIR, exist_ok=True)

# DVS constants from Ish's model
# the static power is basically fixed bias circuitry always on regardless of activity
DVS_P_STATIC = 19.14  # mW
DVS_E_PER_EVENT = 4.95e-9  # J/event tiny cost per brightness change
DVS_REFRACTORY_CAP = 18.75e6  # max events/s before pixel can't keep up

# make all plots look consistent light gray background with subtle grid
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

# blue for CIS, orange for DVS keeps it easy to tell apart
CIS_COLOR = "#1f77b4"
DVS_COLOR = "#ff7f0e"
COLORS_SIZE = {25: "#1f77b4", 50: "#ff7f0e", 100: "#2ca02c"}
SIZE_MARKERS = {25: "o", 50: "s", 100: "D"}
BG_ORDER = ["low_texture", "high_texture"]
BG_LABELS = {
    "low_texture": "Low Texture (plain)",
    "high_texture": "High Texture (cluttered)",
}
VELOCITIES = [10, 50, 100, 200, 500]
OBJ_SIZES = [25, 50, 100]


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def add_finding(ax, text, loc="lower right", fontsize=9):
    """Put a yellow findings box on the plot."""
    box_props = dict(
        boxstyle="round,pad=0.5",
        facecolor="lightyellow",
        edgecolor="#ccaa44",
        alpha=0.92,
        linewidth=1.2,
    )
    anchors = {
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left": (0.03, 0.97, "left", "top"),
        "center right": (0.97, 0.50, "right", "center"),
        "lower center": (0.50, 0.03, "center", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["lower right"])
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=box_props,
    )


def load_data():
    dvs = pd.read_csv(DVS_CSV)
    cis = pd.read_csv(CIS_BG_CSV)
    res_sweep = pd.read_csv(RES_SWEEP_CSV)
    fps_sweep = pd.read_csv(FPS_SWEEP_CSV)

    # merge on the scene parameters that Ramaa's scene model defined for everyone
    # DVS has 3 threshold variants per scene so we get 90 rows (30 CIS x 3 thresholds)
    merge_keys = ["object_size_px", "velocity_px_s", "background"]
    merged = dvs.merge(
        cis[
            merge_keys
            + [
                "power_mW",
                "adc_bits_used",
                "feasible",
                "max_frame_rate_hz",
                "required_fps",
                "dynamic_range_dB",
                "fps_clipped_warning",
            ]
        ],
        on=merge_keys,
        how="left",
        suffixes=("_dvs", "_cis"),
    )
    merged.rename(
        columns={"power_total_mW": "dvs_power_mW", "power_mW": "cis_power_mW"},
        inplace=True,
    )
    # the ratio tells us "how many times more power does CIS use?"
    merged["power_ratio"] = merged["cis_power_mW"] / merged["dvs_power_mW"]
    merged["power_savings_mW"] = merged["cis_power_mW"] - merged["dvs_power_mW"]

    merged.to_csv(os.path.join(OUT_DIR, "merged_cis_dvs_data.csv"), index=False)
    print(f"Merged data: {len(merged)} rows -> merged_cis_dvs_data.csv")
    return merged, cis, res_sweep, fps_sweep


# Plot 1: CIS vs DVS power overlay


def plot1_power_vs_velocity(df):
    # use medium threshold as the "fair" comparison point for DVS
    med = df[df["threshold_name"] == "med_threshold"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, bg in zip(axes, BG_ORDER):
        sub = med[med["background"] == bg]

        for sz in OBJ_SIZES:
            s = sub[sub["object_size_px"] == sz].sort_values("velocity_px_s")

            # shaded gap between CIS and DVS
            ax.fill_between(
                s["velocity_px_s"],
                s["dvs_power_mW"],
                s["cis_power_mW"],
                alpha=0.06,
                color=COLORS_SIZE[sz],
            )
            ax.plot(
                s["velocity_px_s"],
                s["cis_power_mW"],
                marker=SIZE_MARKERS[sz],
                linestyle="-",
                linewidth=2,
                color=COLORS_SIZE[sz],
                markersize=7,
                label=f"CIS obj={sz}px",
            )

            # red X where CIS can't keep up
            infeas = s[s["feasible"] == 0]
            if len(infeas):
                ax.scatter(
                    infeas["velocity_px_s"],
                    infeas["cis_power_mW"],
                    marker="X",
                    s=150,
                    c="red",
                    zorder=6,
                    linewidths=1.5,
                    edgecolors="darkred",
                    label="CIS infeasible" if sz == OBJ_SIZES[0] else "",
                )

            ax.plot(
                s["velocity_px_s"],
                s["dvs_power_mW"],
                marker=SIZE_MARKERS[sz],
                linestyle="--",
                linewidth=1.5,
                color=COLORS_SIZE[sz],
                alpha=0.8,
                markersize=6,
                label=f"DVS obj={sz}px",
            )

        ax.set_title(BG_LABELS[bg], fontsize=13, fontweight="bold")
        ax.set_xlabel("Object Velocity (px/s)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_yticks([20, 50, 100, 150, 200])

        # annotate the gap at v=200 so it's obvious how far apart they are
        mid_idx = sub[(sub["object_size_px"] == 50) & (sub["velocity_px_s"] == 200)]
        if len(mid_idx):
            cis_p = mid_idx["cis_power_mW"].values[0]
            dvs_p = mid_idx["dvs_power_mW"].values[0]
            mid_y = np.sqrt(cis_p * dvs_p)
            ax.annotate(
                "",
                xy=(200, dvs_p),
                xytext=(200, cis_p),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5),
            )
            ax.text(
                215,
                mid_y,
                f"{cis_p/dvs_p:.1f}x\ngap",
                fontsize=8,
                color="gray",
                va="center",
            )

    axes[0].set_ylabel("Power (mW, log scale)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fancybox=True,
    )

    fig.suptitle(
        "CIS vs DVS Power for Object Tracking\n(medium contrast threshold, 480x640 resolution)",
        fontsize=14,
        fontweight="bold",
    )

    add_finding(
        axes[1],
        "FINDING: DVS uses 6-9x less power\n"
        "than CIS across all conditions.\n"
        "The gap is visible at every velocity\n"
        "and widens for smaller objects.",
        loc="center right",
        fontsize=8,
    )

    fig.subplots_adjust(bottom=0.15, top=0.88, wspace=0.08)
    save(fig, "plot1_power_vs_velocity.png")


# Plot 2: how the CIS/DVS ratio changes with velocity


def plot2_power_ratio(df):
    med = df[df["threshold_name"] == "med_threshold"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))

    for bg in BG_ORDER:
        ls = "-" if bg == "low_texture" else "--"
        lw = 2.0 if bg == "low_texture" else 1.5
        for sz in OBJ_SIZES:
            s = med[
                (med["background"] == bg) & (med["object_size_px"] == sz)
            ].sort_values("velocity_px_s")
            bg_short = "low tex" if bg == "low_texture" else "high tex"
            ax.plot(
                s["velocity_px_s"],
                s["power_ratio"],
                marker=SIZE_MARKERS[sz],
                linestyle=ls,
                linewidth=lw,
                color=COLORS_SIZE[sz],
                markersize=7,
                label=f"obj={sz}px, {bg_short}",
            )

    # anything below this line would mean CIS is cheaper spoiler: it never is
    ax.axhline(y=1, color="red", linestyle=":", alpha=0.6, linewidth=1.5)
    ax.text(510, 1.1, "Breakeven\n(ratio=1)", fontsize=8, color="red", va="bottom")

    ax.set_xlabel("Object Velocity (px/s)")
    ax.set_ylabel("Power Ratio (CIS / DVS)")
    ax.set_title(
        "CIS-to-DVS Power Ratio vs Object Velocity\n(ratio > 1 means DVS is more efficient)",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)

    add_finding(
        ax,
        "FINDING: Ratio ranges from 6.2x to 9.2x\n"
        "Smaller objects at high velocity show\n"
        "the largest DVS advantage (9.2x)\n"
        "because CIS must increase FPS more.",
        loc="center right",
        fontsize=8,
    )

    fig.tight_layout()
    save(fig, "plot2_power_ratio.png")


# Plot 3: does background texture change the story?


def plot3_background_sensitivity(df):
    med50 = df[
        (df["threshold_name"] == "med_threshold") & (df["object_size_px"] == 50)
    ].copy()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(VELOCITIES))
    width = 0.18

    for i, bg in enumerate(BG_ORDER):
        sub = med50[med50["background"] == bg].sort_values("velocity_px_s")
        offset = (i - 0.5) * width * 2
        cis_color = "#2166ac" if i == 0 else "#92c5de"
        dvs_color = "#e66101" if i == 0 else "#fdb863"
        ax.bar(
            x + offset - width / 2,
            sub["cis_power_mW"].values,
            width,
            label=f"CIS - {BG_LABELS[bg]}",
            alpha=0.9,
            color=cis_color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar(
            x + offset + width / 2,
            sub["dvs_power_mW"].values,
            width,
            label=f"DVS - {BG_LABELS[bg]}",
            alpha=0.9,
            color=dvs_color,
            edgecolor="white",
            linewidth=0.5,
        )

        for j, (c, d) in enumerate(
            zip(sub["cis_power_mW"].values, sub["dvs_power_mW"].values)
        ):
            ax.text(
                x[j] + offset,
                c + 2.5,
                f"{c/d:.1f}x",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v} px/s" for v in VELOCITIES])
    ax.set_xlabel("Object Velocity")
    ax.set_ylabel("Power (mW)")
    ax.set_title(
        "Background Sensitivity: CIS vs DVS Power\n(obj=50px, medium threshold)",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    add_finding(
        ax,
        "FINDING: Background texture has minimal\n"
        "impact on the CIS/DVS ratio (7.0x vs 6.9x).\n"
        "High texture slightly increases both, but\n"
        "DVS advantage remains stable across scenes.",
        loc="center right",
        fontsize=8,
    )

    fig.tight_layout()
    save(fig, "plot3_background_sensitivity.png")


# Plot 4: does DVS threshold matter?


def plot4_threshold_sensitivity(df):
    sub = df[(df["object_size_px"] == 50) & (df["background"] == "low_texture")].copy()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # theta is the contrast threshold lower = more sensitive = more events = more power
    thresh_styles = {
        "low_threshold": {
            "color": "#d62728",
            "marker": "v",
            "label": "DVS  \u03b8=0.10 (high sensitivity)",
        },
        "med_threshold": {
            "color": "#ff7f0e",
            "marker": "s",
            "label": "DVS  \u03b8=0.20 (medium)",
        },
        "high_threshold": {
            "color": "#2ca02c",
            "marker": "^",
            "label": "DVS  \u03b8=0.40 (low sensitivity)",
        },
    }

    for thr, style in thresh_styles.items():
        s = sub[sub["threshold_name"] == thr].sort_values("velocity_px_s")
        ax.plot(
            s["velocity_px_s"],
            s["dvs_power_mW"],
            marker=style["marker"],
            linestyle="-",
            color=style["color"],
            linewidth=2,
            markersize=7,
            label=style["label"],
        )

    cis_vals = sub.groupby("velocity_px_s")["cis_power_mW"].first()
    ax.plot(
        cis_vals.index,
        cis_vals.values,
        "o-",
        color=CIS_COLOR,
        linewidth=2.5,
        markersize=8,
        label="CIS (8-bit ADC)",
        zorder=5,
    )

    # shade the gap even the worst DVS config is way below CIS
    dvs_max = sub.groupby("velocity_px_s")["dvs_power_mW"].max()
    ax.fill_between(
        cis_vals.index, dvs_max.values, cis_vals.values, alpha=0.08, color="gray"
    )
    ax.text(
        300,
        80,
        "Power gap\n(always > 6x)",
        fontsize=9,
        color="gray",
        ha="center",
        style="italic",
    )

    ax.set_xlabel("Object Velocity (px/s)")
    ax.set_ylabel("Power (mW)")
    ax.set_title(
        "DVS Threshold Sensitivity vs CIS Power\n(obj=50px, low texture background)",
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    add_finding(
        ax,
        "FINDING: Even at the most sensitive\n"
        "DVS threshold (\u03b8=0.10, 2x more events),\n"
        "DVS peaks at 21 mW vs CIS at 158 mW.\n"
        "Threshold choice barely affects DVS power.",
        loc="center right",
        fontsize=8,
    )

    fig.tight_layout()
    save(fig, "plot4_threshold_sensitivity.png")


# Plot 5: what if we shrink the CIS resolution?


def plot5_resolution_sensitivity(res_sweep):
    # can we just use a smaller CIS to beat DVS? turns out... no
    fig, ax = plt.subplots(figsize=(10, 6.5))

    res_labels = [
        f"{int(r)}x{int(c)}" for r, c in zip(res_sweep["rows"], res_sweep["cols"])
    ]
    pixels = res_sweep["total_pixels"]
    cis_power = res_sweep["system_total_power_attr_mW"]

    ax.plot(
        pixels,
        cis_power,
        "o-",
        color=CIS_COLOR,
        linewidth=2.5,
        markersize=10,
        label="CIS Power (10-bit, 10 fps)",
        zorder=5,
    )

    # DVS power band for reference
    ax.axhspan(
        19.14, 28.01, alpha=0.15, color=DVS_COLOR, label="DVS power range (19-28 mW)"
    )
    ax.axhline(y=19.14, color=DVS_COLOR, linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=28.01, color=DVS_COLOR, linestyle="--", alpha=0.5, linewidth=1)

    for i, label in enumerate(res_labels):
        ratio = cis_power.iloc[i] / 23.5  # vs mid-range DVS
        ax.annotate(
            f"{label}\n({ratio:.1f}x vs DVS)",
            (pixels.iloc[i], cis_power.iloc[i]),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
        )

    ax.set_xlabel("Total Pixels")
    ax.set_ylabel("Power (mW)")
    ax.set_title(
        "CIS Power vs Sensor Resolution\n(DVS power range shown for comparison)",
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
        )
    )

    add_finding(
        ax,
        "FINDING: CIS power scales linearly with\n"
        "pixel count. Even at the lowest resolution\n"
        "(240x320 = 77K pixels), CIS at 68 mW is\n"
        "still 2.9x above DVS worst-case (28 mW).",
        loc="center right",
        fontsize=8,
    )

    fig.tight_layout()
    save(fig, "plot5_resolution_sensitivity.png")


# Plot 6: what happens if we push velocity way beyond 500 px/s?


def plot6_saturation_extrapolation(df):
    med = df[
        (df["threshold_name"] == "med_threshold") & (df["object_size_px"] == 50)
    ].copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    # push velocity 10x beyond our measured range to see if curves ever cross
    vel_extrap = np.linspace(10, 5000, 500)

    for bg in BG_ORDER:
        sub = med[med["background"] == bg].sort_values("velocity_px_s")
        ls = "-" if bg == "low_texture" else "--"
        lw = 2.5 if bg == "low_texture" else 2.0
        bg_short = "low tex" if bg == "low_texture" else "high tex"

        # fit a line to the measured points and extrapolate
        coeffs = polyfit(sub["velocity_px_s"], sub["event_rate_scaled"], 1)
        event_rate_extrap = np.clip(
            coeffs[0] + coeffs[1] * vel_extrap, 0, DVS_REFRACTORY_CAP
        )
        dvs_power_extrap = DVS_P_STATIC + event_rate_extrap * DVS_E_PER_EVENT * 1e3

        cis_coeffs = polyfit(sub["velocity_px_s"], sub["cis_power_mW"], 1)
        cis_power_extrap = cis_coeffs[0] + cis_coeffs[1] * vel_extrap

        ax.plot(
            vel_extrap,
            dvs_power_extrap,
            ls,
            color=DVS_COLOR,
            label=f"DVS extrap ({bg_short})",
            linewidth=lw,
        )
        ax.plot(
            vel_extrap,
            cis_power_extrap,
            ls,
            color=CIS_COLOR,
            label=f"CIS extrap ({bg_short})",
            linewidth=lw,
        )

        ax.scatter(
            sub["velocity_px_s"],
            sub["dvs_power_mW"],
            color=DVS_COLOR,
            zorder=5,
            s=60,
            edgecolors="white",
            linewidths=1,
        )
        ax.scatter(
            sub["velocity_px_s"],
            sub["cis_power_mW"],
            color=CIS_COLOR,
            zorder=5,
            s=60,
            edgecolors="white",
            linewidths=1,
        )

    # even if every pixel fires at max rate, DVS tops out here
    dvs_max_power = DVS_P_STATIC + DVS_REFRACTORY_CAP * DVS_E_PER_EVENT * 1e3
    ax.axhline(
        y=dvs_max_power,
        color="red",
        linestyle=":",
        alpha=0.5,
        linewidth=1.5,
        label=f"DVS saturation cap ({dvs_max_power:.0f} mW)",
    )

    ax.axvline(x=500, color="gray", linestyle=":", alpha=0.5, linewidth=1.5)
    ax.annotate(
        "Measured\nrange",
        xy=(500, 30),
        fontsize=8,
        color="gray",
        ha="right",
        va="bottom",
    )
    ax.axvspan(500, 5000, alpha=0.04, color="gray")

    ax.set_xlabel("Object Velocity (px/s)")
    ax.set_ylabel("Power (mW)")
    ax.set_title(
        "Extrapolated CIS vs DVS Power Beyond Measured Range\n(obj=50px, med threshold)",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    add_finding(
        ax,
        "FINDING: Even extrapolating to 5000 px/s,\n"
        "CIS and DVS curves NEVER cross.\n"
        "CIS power grows faster (must increase FPS)\n"
        "while DVS grows slowly (only event rate rises).\n"
        f"DVS saturation cap: {dvs_max_power:.0f} mW (still < CIS baseline).",
        loc="center right",
        fontsize=8,
    )

    fig.tight_layout()
    save(fig, "plot6_saturation_extrapolation.png")


# Plot 7: heatmap of CIS/DVS ratio across all conditions --


def plot7_design_rule_heatmap(df):
    med = df[df["threshold_name"] == "med_threshold"].copy()

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    # keep color scale consistent across both heatmaps
    vmin = med["power_ratio"].min() - 0.3
    vmax = med["power_ratio"].max() + 0.3

    for ax, bg in zip([ax1, ax2], BG_ORDER):
        sub = med[med["background"] == bg]
        pivot = sub.pivot_table(
            index="velocity_px_s",
            columns="object_size_px",
            values="power_ratio",
            aggfunc="mean",
        )
        pivot = pivot.sort_index(ascending=True)

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c}px" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v} px/s" for v in pivot.index])
        ax.set_xlabel("Object Size")
        ax.set_ylabel("Velocity")
        ax.set_title(f"{BG_LABELS[bg]}", fontweight="bold")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                vel = pivot.index[i]
                sz = pivot.columns[j]
                row = sub[(sub["velocity_px_s"] == vel) & (sub["object_size_px"] == sz)]
                cis_p = row["cis_power_mW"].values[0]
                dvs_p = row["dvs_power_mW"].values[0]
                txt_color = "white" if val > 7.5 else "black"
                ax.text(
                    j,
                    i - 0.15,
                    f"{val:.1f}x",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=txt_color,
                )
                ax.text(
                    j,
                    i + 0.25,
                    f"{cis_p:.0f} vs {dvs_p:.0f} mW",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                    alpha=0.8,
                )

        # X on cells where CIS can't meet required FPS
        infeas = sub[sub["feasible"] == 0][
            ["velocity_px_s", "object_size_px"]
        ].drop_duplicates()
        for _, row in infeas.iterrows():
            vel_idx = list(pivot.index).index(row["velocity_px_s"])
            sz_idx = list(pivot.columns).index(row["object_size_px"])
            ax.plot(
                sz_idx,
                vel_idx,
                "X",
                color="red",
                markersize=16,
                markeredgecolor="darkred",
                markeredgewidth=1.5,
                zorder=5,
            )

    fig.colorbar(im, cax=cax, label="CIS/DVS Power Ratio")
    fig.suptitle(
        "CIS/DVS Power Ratio Heatmap  |  Higher = DVS more efficient  |  X = CIS infeasible",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.90)
    save(fig, "plot7_design_rule_heatmap.png")


# Plot 8: where does each sensor's power actually go?


def plot8_power_breakdown(df):
    # shows WHY DVS wins: it's almost all static power, barely any dynamic cost
    med = (
        df[
            (df["threshold_name"] == "med_threshold")
            & (df["object_size_px"] == 50)
            & (df["background"] == "low_texture")
        ]
        .copy()
        .sort_values("velocity_px_s")
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(VELOCITIES))
    width = 0.35

    cis_power = med["cis_power_mW"].values
    dvs_static = med["power_static_mW"].values
    dvs_dynamic = med["power_dynamic_mW"].values

    ax.bar(
        x - width / 2,
        cis_power,
        width,
        label="CIS Total Power",
        color=CIS_COLOR,
        alpha=0.9,
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        dvs_static,
        width,
        label="DVS Static (bias + logic)",
        color="#ff7f0e",
        alpha=0.9,
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        dvs_dynamic,
        width,
        bottom=dvs_static,
        label="DVS Dynamic (events)",
        color="#ffbb78",
        alpha=0.9,
        edgecolor="white",
    )

    for i in range(len(VELOCITIES)):
        ax.text(
            x[i] - width / 2,
            cis_power[i] + 2,
            f"{cis_power[i]:.0f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=CIS_COLOR,
        )
        total_dvs = dvs_static[i] + dvs_dynamic[i]
        ax.text(
            x[i] + width / 2,
            total_dvs + 2,
            f"{total_dvs:.1f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=DVS_COLOR,
        )
        # show how much you'd save by switching white text inside the CIS bar
        savings = cis_power[i] - total_dvs
        pct = (savings / cis_power[i]) * 100
        ax.text(
            x[i] - width / 2,
            cis_power[i] / 2,
            f"-{savings:.0f} mW\n({pct:.0f}%)",
            fontsize=8,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v} px/s" for v in VELOCITIES])
    ax.set_xlabel("Object Velocity")
    ax.set_ylabel("Power (mW)")
    ax.set_title(
        "Power Breakdown: CIS (frame-based) vs DVS (event-based)\n"
        "(obj=50px, low texture, med threshold)",
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    add_finding(
        ax,
        "FINDING: DVS power is 97-99% static (bias circuits).\n"
        "Dynamic event power is negligible (<2 mW).\n"
        "CIS must read all 307K pixels every frame,\n"
        "driving ~133-158 mW through PGA + ADC circuits.",
        loc="center right",
        fontsize=8,
    )

    fig.tight_layout()
    save(fig, "plot8_power_breakdown.png")


# Plot 9: where does CIS fail to keep up?


def plot9_feasibility_map(df):
    # CIS can't always keep up high FPS requirements exceed its readout speed
    med = df[df["threshold_name"] == "med_threshold"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, bg in zip(axes, BG_ORDER):
        sub = med[med["background"] == bg]
        pivot_feasible = sub.pivot_table(
            index="velocity_px_s",
            columns="object_size_px",
            values="feasible",
            aggfunc="first",
        ).sort_index()
        pivot_fps = sub.pivot_table(
            index="velocity_px_s",
            columns="object_size_px",
            values="required_fps",
            aggfunc="first",
        ).sort_index()
        pivot_max = sub.pivot_table(
            index="velocity_px_s",
            columns="object_size_px",
            values="max_frame_rate_hz",
            aggfunc="first",
        ).sort_index()

        colors = np.where(pivot_feasible.values == 1, 0.7, 0.2)
        ax.imshow(colors, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(pivot_feasible.columns)))
        ax.set_xticklabels([f"{c}px" for c in pivot_feasible.columns])
        ax.set_yticks(range(len(pivot_feasible.index)))
        ax.set_yticklabels([f"{v} px/s" for v in pivot_feasible.index])
        ax.set_xlabel("Object Size")
        ax.set_ylabel("Velocity")

        max_fps = pivot_max.values[0, 0]
        ax.set_title(
            f"{BG_LABELS[bg]}\nCIS max FPS: {max_fps:.0f} Hz", fontweight="bold"
        )

        for i in range(len(pivot_feasible.index)):
            for j in range(len(pivot_feasible.columns)):
                req = pivot_fps.values[i, j]
                feas = pivot_feasible.values[i, j]
                status = "CIS OK" if feas else "CIS FAIL"
                txt_color = "darkgreen" if feas else "darkred"
                ax.text(
                    j,
                    i - 0.15,
                    f"Need: {req:.0f} FPS",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=txt_color,
                )
                ax.text(
                    j,
                    i + 0.2,
                    status,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=txt_color,
                )

        ax.text(
            0.5,
            -0.18,
            "DVS: Always feasible (no frame rate constraint)",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color=DVS_COLOR,
            fontweight="bold",
        )

    fig.suptitle(
        "CIS Feasibility Map: Can the sensor track the object?\n"
        "(Green = CIS meets required FPS, Red = CIS cannot keep up)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save(fig, "plot9_feasibility_map.png")


# Plot 10: one-page summary for presentations


def plot10_summary_dashboard(df, stats):
    med = df[df["threshold_name"] == "med_threshold"]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "CIS vs DVS Power Comparison -- Executive Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 3, hspace=0.4, wspace=0.35, left=0.06, right=0.94, top=0.90, bottom=0.08
    )

    # A: show the raw power numbers side by side
    ax_a = fig.add_subplot(gs[0, 0])
    categories = ["CIS\n(min)", "CIS\n(max)", "DVS\n(min)", "DVS\n(max)"]
    values = [
        med["cis_power_mW"].min(),
        med["cis_power_mW"].max(),
        med["dvs_power_mW"].min(),
        med["dvs_power_mW"].max(),
    ]
    colors = [CIS_COLOR, CIS_COLOR, DVS_COLOR, DVS_COLOR]
    alphas = [0.6, 1.0, 0.6, 1.0]
    bars = ax_a.bar(categories, values, color=colors, edgecolor="white")
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    for bar, v in zip(bars, values):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            v + 3,
            f"{v:.0f} mW",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_a.set_ylabel("Power (mW)")
    ax_a.set_title("A. Power Ranges", fontweight="bold")

    # B: does background texture change the story? (not really)
    ax_b = fig.add_subplot(gs[0, 1])
    low_ratio = med[med["background"] == "low_texture"]["power_ratio"].mean()
    high_ratio = med[med["background"] == "high_texture"]["power_ratio"].mean()
    bars = ax_b.barh(
        ["High Texture\n(cluttered)", "Low Texture\n(plain)"],
        [high_ratio, low_ratio],
        color=["#fdb863", "#b2df8a"],
        edgecolor="white",
        height=0.5,
    )
    for bar, v in zip(bars, [high_ratio, low_ratio]):
        ax_b.text(
            v + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}x",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax_b.set_xlabel("CIS/DVS Power Ratio")
    ax_b.set_title("B. Avg Ratio by Background", fontweight="bold")
    ax_b.axvline(x=1, color="red", linestyle=":", alpha=0.4)

    # C: the actual mW you'd save this is what matters for battery life
    ax_c = fig.add_subplot(gs[0, 2])
    sub50 = med[(med["object_size_px"] == 50) & (med["background"] == "low_texture")]
    sub50 = sub50.sort_values("velocity_px_s")
    savings = sub50["cis_power_mW"].values - sub50["dvs_power_mW"].values
    ax_c.fill_between(sub50["velocity_px_s"], 0, savings, alpha=0.3, color="#2ca02c")
    ax_c.plot(
        sub50["velocity_px_s"],
        savings,
        "o-",
        color="#2ca02c",
        linewidth=2,
        markersize=6,
    )
    for v, s in zip(sub50["velocity_px_s"], savings):
        ax_c.text(v, s + 2, f"{s:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax_c.set_xlabel("Velocity (px/s)")
    ax_c.set_ylabel("Power Saved (mW)")
    ax_c.set_title("C. Power Savings Using DVS", fontweight="bold")

    # D: compact version of the full heatmap for the summary page
    ax_d = fig.add_subplot(gs[1, 0])
    sub_low = med[med["background"] == "low_texture"]
    pivot = sub_low.pivot_table(
        index="velocity_px_s",
        columns="object_size_px",
        values="power_ratio",
        aggfunc="mean",
    ).sort_index()
    im = ax_d.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=6, vmax=10)
    ax_d.set_xticks(range(len(pivot.columns)))
    ax_d.set_xticklabels([f"{c}px" for c in pivot.columns])
    ax_d.set_yticks(range(len(pivot.index)))
    ax_d.set_yticklabels([f"{v}" for v in pivot.index])
    ax_d.set_xlabel("Object Size")
    ax_d.set_ylabel("Velocity (px/s)")
    ax_d.set_title("D. Ratio Heatmap (low tex)", fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax_d.text(
                j,
                i,
                f"{pivot.values[i,j]:.1f}x",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if pivot.values[i, j] > 7.5 else "black",
            )

    # E: CIS can't even do the job in some cases DVS always can
    ax_e = fig.add_subplot(gs[1, 1])
    n_total = len(
        med[["object_size_px", "velocity_px_s", "background"]].drop_duplicates()
    )
    n_infeas = len(
        med[med["feasible"] == 0][
            ["object_size_px", "velocity_px_s", "background"]
        ].drop_duplicates()
    )
    n_feas = n_total - n_infeas
    wedges, texts, autotexts = ax_e.pie(
        [n_feas, n_infeas],
        labels=["CIS feasible", "CIS infeasible"],
        colors=["#b2df8a", "#fb9a99"],
        autopct=lambda pct: f"{int(pct/100 * n_total)}/{n_total}",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax_e.set_title("E. CIS Feasibility\n(DVS: 30/30 feasible)", fontweight="bold")

    # F: the bottom line what an engineer should take away from all this
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    rule_text = (
        "DESIGN RULE\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"DVS is ALWAYS preferred\n"
        f"for power-efficient tracking.\n\n"
        f"Power advantage: {stats['min_ratio']:.1f}x -- {stats['max_ratio']:.1f}x\n"
        f"(mean {stats['mean_ratio']:.1f}x less power)\n\n"
        "No crossover exists:\n"
        "CIS gap widens with velocity.\n\n"
        "Use CIS only when full-frame\n"
        "RGB capture is required."
    )
    ax_f.text(
        0.5,
        0.5,
        rule_text,
        transform=ax_f.transAxes,
        fontsize=11,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="lightyellow",
            edgecolor="#ccaa44",
            linewidth=2,
        ),
        family="monospace",
        linespacing=1.4,
    )
    ax_f.set_title("F. Design Rule", fontweight="bold")

    save(fig, "plot10_summary_dashboard.png")


# Crossover analysis


def crossover_analysis(df, res_sweep):
    med = df[df["threshold_name"] == "med_threshold"]

    ratios = med["power_ratio"]
    min_ratio = ratios.min()
    max_ratio = ratios.max()
    mean_ratio = ratios.mean()
    min_row = med.loc[ratios.idxmin()]
    max_row = med.loc[ratios.idxmax()]

    # try to find where CIS and DVS power curves would cross (they don't)
    sub = med[
        (med["object_size_px"] == 50) & (med["background"] == "low_texture")
    ].sort_values("velocity_px_s")

    er_coeffs = polyfit(sub["velocity_px_s"], sub["event_rate_scaled"], 1)
    cis_coeffs = polyfit(sub["velocity_px_s"], sub["cis_power_mW"], 1)

    dvs_a = DVS_P_STATIC + er_coeffs[0] * DVS_E_PER_EVENT * 1e3
    dvs_b = er_coeffs[1] * DVS_E_PER_EVENT * 1e3
    denom = cis_coeffs[1] - dvs_b
    crossover_v = (
        (dvs_a - cis_coeffs[0]) / denom if abs(denom) > 1e-12 else float("inf")
    )

    # what if we shrink the CIS sensor until its power matches DVS?
    res_coeffs = polyfit(
        res_sweep["total_pixels"], res_sweep["system_total_power_attr_mW"], 1
    )
    dvs_max = 28.01
    crossover_pixels = (
        (dvs_max - res_coeffs[0]) / res_coeffs[1]
        if abs(res_coeffs[1]) > 1e-12
        else float("inf")
    )

    lines = [
        "=" * 65,
        "CROSSOVER ANALYSIS: CIS vs DVS Power",
        "=" * 65,
        "",
        "POWER RATIO STATISTICS (CIS / DVS, med threshold):",
        f"  Minimum ratio:  {min_ratio:.2f}x",
        f"    at: obj={int(min_row['object_size_px'])}px, vel={int(min_row['velocity_px_s'])}px/s, bg={min_row['background']}",
        f"  Maximum ratio:  {max_ratio:.2f}x",
        f"    at: obj={int(max_row['object_size_px'])}px, vel={int(max_row['velocity_px_s'])}px/s, bg={max_row['background']}",
        f"  Mean ratio:     {mean_ratio:.2f}x",
        "",
        "CROSSOVER POINT (obj=50px, low texture, med threshold):",
    ]
    if crossover_v < 0:
        lines += [
            "  No crossover exists.",
            "  CIS power grows FASTER with velocity than DVS power,",
            "  so the gap widens as scenes become more dynamic.",
            "  The two power curves diverge -- DVS never catches CIS.",
        ]
    elif crossover_v > 500:
        lines += [
            f"  Hypothetical crossover velocity: {crossover_v:.0f} px/s",
            f"  >> This is {crossover_v/500:.0f}x beyond the measured range (500 px/s)",
            "  >> Physically unrealistic for a 640x480 sensor",
        ]
    else:
        lines.append(f"  Crossover velocity: {crossover_v:.0f} px/s")

    lines += ["", "RESOLUTION CROSSOVER (CIS power = DVS worst-case 28 mW):"]
    if crossover_pixels > 0:
        eq_h = int(np.sqrt(crossover_pixels * 3 / 4))
        eq_w = int(np.sqrt(crossover_pixels * 4 / 3))
        lines += [
            f"  Hypothetical pixel count: {crossover_pixels:.0f} pixels",
            f"  Equivalent resolution:    ~{eq_h}x{eq_w} (4:3 ratio)",
            "  >> Impractically low for any tracking application",
        ]
    else:
        lines += [
            f"  Hypothetical pixel count: {crossover_pixels:.0f} (negative)",
            "  >> CIS power never drops to DVS level at any resolution",
        ]

    lines += [
        "",
        "CONCLUSION:",
        "  No crossover exists in the measured parameter space.",
        f"  DVS is always more power-efficient by at least {min_ratio:.1f}x.",
        "=" * 65,
    ]

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "crossover_analysis.txt"), "w") as f:
        f.write(text)
    print(f"\n{text}\n")
    return {
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "mean_ratio": mean_ratio,
        "crossover_velocity": crossover_v,
        "crossover_pixels": crossover_pixels,
    }


# Design rule text output


def design_rule(df, stats):
    med = df[df["threshold_name"] == "med_threshold"]
    low_bg_mean = med[med["background"] == "low_texture"]["power_ratio"].mean()
    high_bg_mean = med[med["background"] == "high_texture"]["power_ratio"].mean()

    infeas = med[med["feasible"] == 0]
    infeas_desc = ""
    if len(infeas):
        infeas_cases = infeas[
            ["object_size_px", "velocity_px_s", "background"]
        ].drop_duplicates()
        infeas_desc = "\n".join(
            f"    - obj={int(r['object_size_px'])}px, vel={int(r['velocity_px_s'])}px/s, bg={r['background']}"
            for _, r in infeas_cases.iterrows()
        )

    text = f"""{'='*65}
DESIGN RULE: CIS vs DVS for Object Tracking
{'='*65}

Task: Object tracking at 480x640 resolution
Objects: 25-100 px, moving at 10-500 px/s
Backgrounds: Low texture (5% edge density) and High texture (40%)

RULE:
  DVS is ALWAYS more power-efficient for this tracking task,
  consuming {stats['min_ratio']:.1f}x to {stats['max_ratio']:.1f}x less power than CIS
  (mean: {stats['mean_ratio']:.1f}x across all conditions).

DETAILS:
  - Low texture background:  avg ratio = {low_bg_mean:.1f}x (DVS advantage)
  - High texture background: avg ratio = {high_bg_mean:.1f}x (DVS advantage)
  - DVS power range:  19.1 - 28.0 mW (dominated by static power)
  - CIS power range:  133 - 190 mW (dominated by readout circuits)

CIS INFEASIBILITY:
  CIS cannot meet tracking requirements at these operating points:
{infeas_desc}
  (required FPS exceeds max frame rate of sensor)
  DVS has NO infeasibility -- it is always able to track.

WHEN TO USE EACH SENSOR:
  Use DVS when:
    - Power efficiency is the primary design objective
    - Scene has moderate to low activity (background edge density < 40%)
    - Object velocities are in the 10-500 px/s range

  Use CIS when:
    - Full-frame RGB image capture is required (classification, recognition)
    - Scene documentation is needed (not just motion detection)
    - Color information is essential for the downstream task
    - Power budget is not the primary constraint

CROSSOVER CONDITIONS:
  No crossover exists in the measured or extrapolated parameter space.
  CIS power grows faster with velocity than DVS, so the gap WIDENS
  as scenes become more dynamic. DVS dominance is fundamental:
  CIS must read out all pixels every frame, while DVS only responds
  to brightness changes.

{'='*65}
"""
    with open(os.path.join(OUT_DIR, "design_rule_summary.txt"), "w") as f:
        f.write(text)
    print(text)


def main():
    # load everything, merge CIS + DVS on shared scene params, then generate all outputs
    merged, cis, res_sweep, fps_sweep = load_data()
    plot1_power_vs_velocity(merged)
    plot2_power_ratio(merged)
    plot3_background_sensitivity(merged)
    plot4_threshold_sensitivity(merged)
    plot5_resolution_sensitivity(res_sweep)
    plot6_saturation_extrapolation(merged)
    plot7_design_rule_heatmap(merged)
    plot8_power_breakdown(merged)
    plot9_feasibility_map(merged)
    stats = crossover_analysis(merged, res_sweep)
    plot10_summary_dashboard(merged, stats)
    design_rule(merged, stats)


if __name__ == "__main__":
    main()
