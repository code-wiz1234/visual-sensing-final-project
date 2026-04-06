"""Overlay: synthetic baseline vs empirical results side by side.

Both datasets use the same crossover CSV schema (sensor_name, sensor_type,
tracker, velocity_scale, mean_velocity_px_s, power_mW, mota, idf1, ...).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_CSV = os.path.join(THIS_DIR, "synthetic_crossover_results.csv")
EMPIRICAL_CSV = os.path.join(THIS_DIR, "real_world_results.csv")
CROSSOVER_CSV = os.path.join(THIS_DIR, "crossover_results.csv")
SLIDES_DIR = os.path.join(THIS_DIR, "real_world_slides")
os.makedirs(SLIDES_DIR, exist_ok=True)

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
DVS_COLORS = {
    "DVS: Lichtsteiner 2008": "#1b9e77", "DVS: DAVIS346": "#d95f02",
    "DVS: Samsung Gen3.1": "#7570b3", "DVS: Prophesee IMX636": "#e7298a",
}
CIS_COLORS = {
    "CIS: OV7251 640x480 10b": "#2196F3", "CIS: IMX327 1080p 12b": "#FF5722",
    "CIS: AR0234 1200p 10b": "#4CAF50", "CIS: IMX462 1080p 12b": "#9C27B0",
}
ALL_COLORS = {**DVS_COLORS, **CIS_COLORS}


def _relabel(df):
    df = df.copy()
    df["sensor_name"] = df["sensor_name"].map(lambda sensor_name: _DISPLAY_NAMES.get(sensor_name, sensor_name))
    return df


def _save(fig, name):
    path = os.path.join(SLIDES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def _plot_crossover_panel(ax, df, title):
    """Plot MOTA vs velocity for all sensors in a crossover dataframe."""
    sort_df = df[df["tracker"] == "sort"] if "tracker" in df.columns else df
    for sensor_name in sorted(sort_df["sensor_name"].unique()):
        grp = sort_df[sort_df["sensor_name"] == sensor_name].sort_values("mean_velocity_px_s")
        color = ALL_COLORS.get(sensor_name, "gray")
        is_dvs = grp.iloc[0]["sensor_type"] == "DVS"
        marker = "^" if is_dvs else "s"
        style = "-" if is_dvs else "--"
        ax.plot(grp["mean_velocity_px_s"], grp["mota"],
                marker=marker, linestyle=style, color=color,
                linewidth=1.5, label=sensor_name)
    ax.set(xlabel="Velocity (px/s)", ylabel="MOTA", title=title)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_09_synthetic_vs_empirical(synthetic, crossover):
    """Side-by-side: synthetic MOTA vs empirical MOTA.

    Both use the same schema, so we can plot them with the same function.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if synthetic is not None:
        _plot_crossover_panel(axes[0], synthetic,
                              "Synthetic Scenes\n(generated trajectories, SORT)")
    else:
        axes[0].text(0.5, 0.5, "No synthetic data.\nRun tracking_baseline.py",
                     transform=axes[0].transAxes, ha="center", va="center")
        axes[0].set_title("Synthetic (not available)")

    if crossover is not None:
        _plot_crossover_panel(axes[1], crossover,
                              "Real-World MOT17\n(video frames, SORT)")
    else:
        axes[1].text(0.5, 0.5, "No crossover data.\nRun run_crossover.py",
                     transform=axes[1].transAxes, ha="center", va="center")
        axes[1].set_title("Real-World (not available)")

    fig.suptitle("Same Pipeline, Different Source: Synthetic vs Real-World",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "09_synthetic_vs_empirical.png")


def plot_10_color_ablation(empirical):
    """CIS-RGB vs CIS-gray ablation from empirical benchmark data."""
    cis = empirical[empirical["sensor"].isin(["CIS-RGB", "CIS-gray"])]
    if cis.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(
        axes, ["mota", "idf1", "id_switches"],
        ["MOTA (higher better)", "IDF1 (higher better)", "ID switches (lower better)"],
    ):
        pivot = cis.pivot_table(index="sequence", columns="sensor", values=metric)
        pivot.plot.bar(ax=ax, color={"CIS-RGB": "#D35400", "CIS-gray": "#888888"})
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.set_xlabel("")

    plt.suptitle("Color (CIS-RGB) vs Monochrome (CIS-gray), same tracker",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_color_ablation.png")


def main():
    synthetic = None
    if os.path.exists(SYNTHETIC_CSV):
        synthetic = _relabel(pd.read_csv(SYNTHETIC_CSV))
        print(f"Synthetic: {len(synthetic)} rows")

    crossover = None
    if os.path.exists(CROSSOVER_CSV):
        crossover = _relabel(pd.read_csv(CROSSOVER_CSV))
        print(f"Crossover: {len(crossover)} rows")

    empirical = None
    if os.path.exists(EMPIRICAL_CSV):
        empirical = pd.read_csv(EMPIRICAL_CSV)
        print(f"Empirical: {len(empirical)} rows")

    if synthetic is not None or crossover is not None:
        plot_09_synthetic_vs_empirical(synthetic, crossover)

    if empirical is not None:
        plot_10_color_ablation(empirical)

    if synthetic is None and crossover is None and empirical is None:
        print("No data found. Run tracking_baseline.py and/or run_crossover.py first.")


if __name__ == "__main__":
    main()
