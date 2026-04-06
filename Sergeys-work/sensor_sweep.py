"""Sweep commercial sensors across velocities using measured MOT17 workload.

Output: sensor_sweep_results.csv, sensor_sweep.png
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..", "Ishs-work")))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..", "Remaas-work")))

from sensor_database import DVS_SENSORS, CIS_SENSORS, print_sensor_table
from visualcomputing import compute_event_rate, compute_fps_min, object_sizes, backgrounds, false_pos_rate


def main():
    print_sensor_table()
    print()

    mot17_event_rate = 762_628  # from our real run at theta=0.20
    mot17_fps = 30.0

    velocity_sweep = [10, 50, 100, 200, 500, 1000, 2000]
    object_size = 50
    background_density = 0.05  # low_texture

    rows = []

    for velocity in velocity_sweep:
        event_rate = compute_event_rate(velocity, object_size, background_density, false_pos_rate)
        required_fps = compute_fps_min(velocity, object_size, safety=10)

        for dvs in DVS_SENSORS:
            p = dvs.power_mw(event_rate)
            pe = dvs.position_error_px(velocity)
            rows.append({
                "velocity_px_s": velocity,
                "sensor_type": "DVS",
                "sensor_name": dvs.name,
                "resolution": f"{dvs.resolution[0]}x{dvs.resolution[1]}",
                "event_rate": event_rate,
                "required_fps": None,
                "power_mW": round(p, 3),
                "position_error_px": round(pe, 6),
                "price": dvs.price_usd,
                "pixel_latency_us": dvs.pixel_latency_us,
                "e_per_event_nj": dvs.e_per_event_nj,
            })

        for cis in CIS_SENSORS:
            p = cis.power_mw(required_fps)
            actual_fps = min(required_fps, cis.max_fps)
            pe = 1.5 * velocity / actual_fps if actual_fps > 0 else float("inf")
            rows.append({
                "velocity_px_s": velocity,
                "sensor_type": "CIS",
                "sensor_name": cis.name,
                "resolution": f"{cis.resolution[0]}x{cis.resolution[1]}",
                "event_rate": None,
                "required_fps": required_fps,
                "power_mW": round(p, 3),
                "position_error_px": round(pe, 4),
                "price": cis.price_usd,
                "pixel_latency_us": None,
                "e_per_event_nj": None,
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(THIS_DIR, "sensor_sweep_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")

    print("\n" + "=" * 80)
    print("POWER ON MOT17-04 WORKLOAD (762k ev/s, 30 fps)")
    print("=" * 80)
    print(f"\n{'Sensor':<25s} {'Type':>4s} {'Power':>9s} {'Price':>20s}")
    print("-" * 65)
    for dvs in DVS_SENSORS:
        p = dvs.power_mw(mot17_event_rate)
        print(f"{dvs.name:<25s} {'DVS':>4s} {p:>7.1f}mW {dvs.price_usd:>20s}")
    for cis in CIS_SENSORS:
        p = cis.power_mw(mot17_fps)
        print(f"{cis.name:<25s} {'CIS':>4s} {p:>7.1f}mW {cis.price_usd:>20s}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for dvs in DVS_SENSORS:
        sensor_data = df[(df.sensor_name == dvs.name)]
        ax1.plot(sensor_data.velocity_px_s, sensor_data.power_mW, marker="o", label=dvs.name, linewidth=2)
    ax1.set_xlabel("Object Velocity (px/s)")
    ax1.set_ylabel("DVS Power (mW)")
    ax1.set_title("DVS: Power vs Velocity by Sensor")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    for cis in CIS_SENSORS:
        sensor_data = df[(df.sensor_name == cis.name)]
        ax2.plot(sensor_data.velocity_px_s, sensor_data.power_mW, marker="s", label=cis.name, linewidth=2)
    ax2.set_xlabel("Object Velocity (px/s)")
    ax2.set_ylabel("CIS Power (mW)")
    ax2.set_title("CIS: Power vs Velocity by Sensor")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Commercial Sensor Power vs Object Velocity\n"
                 "(DVS scales with event rate; CIS scales with required FPS)",
                 fontweight="bold")
    plt.tight_layout()
    out_png = os.path.join(THIS_DIR, "sensor_sweep.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"\nSaved {out_png}")

    fig, ax = plt.subplots(figsize=(10, 6))

    def _price_mid(price_str):
        import re
        nums = re.findall(r'[\d,]+', price_str.replace(",", ""))
        if not nums:
            return None
        vals = [float(n) for n in nums]
        return sum(vals) / len(vals)

    for dvs in DVS_SENSORS:
        sensor_power = dvs.power_mw(mot17_event_rate)
        sensor_price = _price_mid(dvs.price_usd)
        if sensor_price:
            ax.scatter(sensor_price, sensor_power, s=120, marker="^", zorder=5)
            ax.annotate(dvs.name, (sensor_price, sensor_power), fontsize=8,
                        textcoords="offset points", xytext=(5, 5))
    for cis in CIS_SENSORS:
        sensor_power = cis.power_mw(mot17_fps)
        sensor_price = _price_mid(cis.price_usd)
        if sensor_price:
            ax.scatter(sensor_price, sensor_power, s=120, marker="o", zorder=5)
            ax.annotate(cis.name, (sensor_price, sensor_power), fontsize=8,
                        textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Approximate Price (USD)")
    ax.set_ylabel("Power on MOT17 workload (mW)")
    ax.set_title("Sensor Power vs Price\n(MOT17-04: 762k ev/s DVS, 30 fps CIS)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(["DVS sensors (^)", "CIS sensors (o)"], fontsize=10)
    plt.tight_layout()
    out_price = os.path.join(THIS_DIR, "sensor_price_vs_power.png")
    plt.savefig(out_price, dpi=200)
    plt.close()
    print(f"Saved {out_price}")


if __name__ == "__main__":
    main()
