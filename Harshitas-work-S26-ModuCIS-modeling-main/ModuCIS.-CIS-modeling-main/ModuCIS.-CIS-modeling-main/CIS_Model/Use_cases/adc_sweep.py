# adc_sweep.py
# Run from Use_cases folder:
#   python3 adc_sweep.py
#
# Outputs:
#   - adc_sweep_detailed.csv
#   - adc_sweep_full_output.csv
#   - power_vs_adc_bits.png

import sys
import os
import csv
import re
import io
from contextlib import redirect_stdout

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from Top_10_22_CNN_optical import CIS_Array


# -----------------------------
# 1) Sweep settings (ADC bits sweep)
# -----------------------------
fps_fixed = 10
rows, cols = 480, 640
adc_bits_list = [8, 10, 12]

csv_detailed = "adc_sweep_detailed.csv"
csv_full = "adc_sweep_full_output.csv"
png_name = "power_vs_adc_bits.png"


# -----------------------------
# 2) Fixed configuration
# -----------------------------
PIXEL_BINNING = [1, 1]
INPUT_CLK_FREQ = 20e6
PGA_DC_GAIN = 17.38

INPUT_PIXEL_MAP = [
    [0.2, 1.75, 1.75],
    [0.2, 1.75, 1.75],
    [0.2, 1.75, 1.75],
    [0.2, 1.75, 1.75],
    [0.2, 1.75, 1.75],
]


def _find_float(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_metrics(out: str) -> dict:
    metrics = {}

    metrics["effective_pixels"] = _find_float(r"Num of Effective pixels\s*:\s*([0-9.]+)", out)
    metrics["frame_time_ms"] = _find_float(r"Frame Time\s*:\s*([0-9.]+)\s*ms", out)
    metrics["adc_time_us"] = _find_float(r"ADC Time\s*:\s*([0-9.]+)\s*us", out)
    metrics["max_frame_rate_hz"] = _find_float(r"Max Frame Rate\s*:\s*([0-9.]+)\s*Hz", out)

    total_mw = _find_float(r"System Total Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["system_total_power_mW_printed"] = total_mw

    metrics["pga_power_mW"] = _find_float(r"PGA Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["mipi_power_mW"] = _find_float(r"MIPI Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["ss_adc_power_mW"] = _find_float(r"SS ADC Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["sar_adc_power_mW"] = _find_float(r"SAR ADC Power\s*:\s*([0-9.]+)\s*mW", out)

    metrics["dynamic_range_dB_first"] = _find_float(
        r"0th Color,\s*0th pixel size\s*:\s*([0-9.]+)\s*dB", out
    )
    metrics["total_noise_e"] = _find_float(r"Total Noise\s*:\s*([0-9.]+)\s*e-", out)

    metrics["fps_clipped_warning"] = 1 if "Input frame rate is too high" in out else 0

    return metrics


detailed_rows = []
full_output_rows = []

plot_bits = []
plot_power_mW = []

print("\n=== ADC Bits Sweep: Power vs ADC Resolution ===")
print(f"Fixed: resolution={rows}x{cols}, fps={fps_fixed}\n")

for bits in adc_bits_list:
    buf = io.StringIO()

    with redirect_stdout(buf):
        sensor = CIS_Array(
            feature_size_nm=65,
            analog_V_dd=2.8,
            digital_V_dd=0.8,
            input_clk_freq=INPUT_CLK_FREQ,

            photodiode_type=0,
            CFA_type=0,
            input_pixel_map=INPUT_PIXEL_MAP,
            pd_E=10,
            PD_saturation_level=0.6,

            num_rows=int(rows),
            num_cols=int(cols),
            Pixel_type=0,
            pixel_binning_map=PIXEL_BINNING,
            num_PD_per_tap=1,
            num_of_tap=1,
            num_of_unused_tap=0,

            frame_rate=float(fps_fixed),
            max_subframe_rates=1,
            subframe_rates=1,
            exposure_time=0,

            IO_type=1,
            MUX_type=0,
            PGA_type=1,
            CDS_type=0,
            CIS_type=0,
            ADC_type=1,

            adc_resolution=int(bits),
            PGA_DC_gain=PGA_DC_GAIN,
            num_mux_input=1,
            CDS_amp_gain=8,
            bias_voltage=0.8,
            comparator_bias_voltage=0.8,
            ADC_input_clk_freq=22e6,

            if_PLL=1,
            if_time_generator=1,
            PLL_output_frequency=44e6,

            additional_latency=0,
            CNN_kernel=3
        )

        print("Total Power (sensor.system_total_power):", sensor.system_total_power)

    out = buf.getvalue()
    metrics = parse_metrics(out)

    power_W = float(sensor.system_total_power)
    power_mW_attr = power_W * 1000.0
    p_mW = power_mW_attr if power_mW_attr is not None else metrics["system_total_power_mW_printed"]

    print(f"ADC {bits:>2}-bit | Power ≈ {p_mW:.2f} mW | DR ≈ {metrics.get('dynamic_range_dB_first', None)} dB")

    metrics["rows"] = rows
    metrics["cols"] = cols
    metrics["total_pixels"] = rows * cols
    metrics["fps_requested"] = fps_fixed
    metrics["adc_bits"] = bits
    metrics["system_total_power_attr_W"] = power_W
    metrics["system_total_power_attr_mW"] = power_mW_attr

    detailed_rows.append(metrics)
    full_output_rows.append({
        "fps_requested": fps_fixed,
        "rows": rows,
        "cols": cols,
        "adc_bits": bits,
        "full_output": out
    })

    if p_mW is not None:
        plot_bits.append(bits)
        plot_power_mW.append(p_mW)

print("\n✅ Sweep complete.\n")


fieldnames = [
    "fps_requested", "rows", "cols", "total_pixels", "adc_bits",
    "fps_clipped_warning",
    "effective_pixels",
    "frame_time_ms", "adc_time_us", "max_frame_rate_hz",
    "system_total_power_attr_W", "system_total_power_attr_mW",
    "system_total_power_mW_printed",
    "pga_power_mW", "mipi_power_mW",
    "sar_adc_power_mW", "ss_adc_power_mW",
    "dynamic_range_dB_first", "total_noise_e"
]

with open(csv_detailed, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in detailed_rows:
        writer.writerow({k: row.get(k, None) for k in fieldnames})

print(f"Saved detailed CSV: {csv_detailed}")


with open(csv_full, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fps_requested", "rows", "cols", "adc_bits", "full_output"])
    writer.writeheader()
    for row in full_output_rows:
        writer.writerow(row)

print(f"Saved full output CSV (raw text): {csv_full}")


if len(plot_bits) >= 2:
    plt.figure()
    plt.plot(plot_bits, plot_power_mW, marker="o")
    plt.xlabel("ADC Resolution (bits)")
    plt.ylabel("Total Power (mW)")
    plt.title(f"Power vs ADC Bits (Resolution {rows}x{cols}, FPS={fps_fixed})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_name, dpi=200)
    print(f"Saved plot: {png_name}")
else:
    print("Not enough valid points to plot (need >=2).")
