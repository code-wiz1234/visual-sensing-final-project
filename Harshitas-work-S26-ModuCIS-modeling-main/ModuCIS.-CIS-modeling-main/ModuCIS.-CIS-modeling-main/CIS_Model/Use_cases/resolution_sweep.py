# resolution_sweep.py
# Run from Use_cases folder:
#   python3 resolution_sweep.py
#
# Outputs:
#   - res_sweep_detailed.csv (structured metrics)
#   - res_sweep_full_output.csv (raw printout per run)
#   - power_vs_resolution.png (graph)

import sys
import os
import csv
import re
import io
from contextlib import redirect_stdout

# allow imports from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from Top_10_22_CNN_optical import CIS_Array


# -----------------------------
# 1) Sweep settings (Resolution Sweep)
# -----------------------------
fps_fixed = 10
adc_bits = 10

# (rows, cols)
resolutions = [
    (240, 320),
    (480, 640),
    (720, 1280),
    (1080, 1920),
]

csv_detailed = "res_sweep_detailed.csv"
csv_full = "res_sweep_full_output.csv"
png_name = "power_vs_resolution.png"


# -----------------------------
# 2) Fixed configuration (same as your Use_case_1 style)
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


# -----------------------------
# 3) Regex helpers to parse printed output
# -----------------------------
def _find_float(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_metrics(out: str) -> dict:
    """
    Parse key metrics from the model's printed output.
    If a metric is not found, value = None.
    """
    metrics = {}

    # Timing
    metrics["effective_pixels"] = _find_float(r"Num of Effective pixels\s*:\s*([0-9.]+)", out)
    metrics["frame_time_ms"] = _find_float(r"Frame Time\s*:\s*([0-9.]+)\s*ms", out)
    metrics["exposure_time_us"] = _find_float(r"Exposure Time\s*:\s*([0-9.]+)\s*us", out)
    metrics["readout_time_us"] = _find_float(r"Readout Time\s*:\s*([0-9.]+)\s*us", out)
    metrics["io_time_us"] = _find_float(r"I/O Time\s*:\s*([0-9.]+)\s*us", out)
    metrics["adc_time_us"] = _find_float(r"ADC Time\s*:\s*([0-9.]+)\s*us", out)
    metrics["idle_time_us"] = _find_float(r"Idle Time\s*:\s*([0-9.]+)\s*us", out)

    metrics["frame_rate_hz_printed"] = _find_float(r"Frame Rate\s*:\s*([0-9.]+)\s*HZ", out)
    metrics["max_frame_rate_hz"] = _find_float(r"Max Frame Rate\s*:\s*([0-9.]+)\s*Hz", out)

    # Total printed power (mW)
    total_mw = _find_float(r"System Total Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["system_total_power_mW_printed"] = total_mw
    metrics["system_total_power_W_printed"] = (total_mw / 1000.0) if total_mw is not None else None

    # Block powers (mW) – if present in printout
    metrics["pixel_array_power_mW"] = _find_float(r"Pixel Array Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["readout_circuit_power_mW"] = _find_float(r"Readout Circuit Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["pga_power_mW"] = _find_float(r"PGA Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["ss_adc_power_mW"] = _find_float(r"SS ADC Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["sar_adc_power_mW"] = _find_float(r"SAR ADC Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["mipi_power_mW"] = _find_float(r"MIPI Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["pll_power_mW"] = _find_float(r"Phase Locked Loop Power\s*:\s*([0-9.]+)\s*mW", out)

    # DR / noise (optional)
    metrics["dynamic_range_dB_first"] = _find_float(
        r"0th Color,\s*0th pixel size\s*:\s*([0-9.]+)\s*dB", out
    )
    metrics["total_noise_e"] = _find_float(r"Total Noise\s*:\s*([0-9.]+)\s*e-", out)

    # Warning if model clipped
    metrics["fps_clipped_warning"] = 1 if "Input frame rate is too high" in out else 0

    return metrics


# -----------------------------
# 4) Run sweep + capture outputs
# -----------------------------
detailed_rows = []
full_output_rows = []

plot_x_pixels = []
plot_power_mW = []

print("\n=== Resolution Sweep: Power vs Resolution ===")
print(f"Fixed: fps={fps_fixed}, adc_bits={adc_bits}\n")

for (rows, cols) in resolutions:

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

            adc_resolution=int(adc_bits),
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

        # keep this in raw output
        print("Total Power (sensor.system_total_power):", sensor.system_total_power)

    out = buf.getvalue()
    metrics = parse_metrics(out)

    # Attribute total power
    power_W = None
    try:
        power_W = float(sensor.system_total_power)
    except Exception:
        power_W = None

    power_mW_attr = (power_W * 1000.0) if power_W is not None else None

    # Choose power for plotting/summary: prefer attribute
    p_mW = power_mW_attr if power_mW_attr is not None else metrics["system_total_power_mW_printed"]

    num_px = rows * cols

    # Console summary
    if p_mW is not None:
        print(f"Res {rows}x{cols} ({num_px}) | Power ≈ {p_mW:.2f} mW")
        plot_x_pixels.append(num_px)
        plot_power_mW.append(p_mW)
    else:
        print(f"Res {rows}x{cols} ({num_px}) | Power: N/A")

    # Save detailed metrics
    metrics["fps_requested"] = fps_fixed
    metrics["rows"] = rows
    metrics["cols"] = cols
    metrics["adc_bits"] = adc_bits
    metrics["total_pixels"] = num_px
    metrics["system_total_power_attr_W"] = power_W
    metrics["system_total_power_attr_mW"] = power_mW_attr

    detailed_rows.append(metrics)

    # Save raw output
    full_output_rows.append({
        "fps_requested": fps_fixed,
        "rows": rows,
        "cols": cols,
        "adc_bits": adc_bits,
        "full_output": out
    })

print("\n✅ Sweep complete.\n")


# -----------------------------
# 5) Write detailed CSV
# -----------------------------
fieldnames = [
    "fps_requested", "rows", "cols", "total_pixels", "adc_bits",
    "fps_clipped_warning",
    "effective_pixels",
    "frame_time_ms", "exposure_time_us", "readout_time_us", "io_time_us", "adc_time_us", "idle_time_us",
    "frame_rate_hz_printed", "max_frame_rate_hz",
    "system_total_power_attr_W", "system_total_power_attr_mW",
    "system_total_power_mW_printed", "system_total_power_W_printed",
    "pixel_array_power_mW", "readout_circuit_power_mW", "pga_power_mW",
    "sar_adc_power_mW", "ss_adc_power_mW", "mipi_power_mW", "pll_power_mW",
    "dynamic_range_dB_first", "total_noise_e"
]

with open(csv_detailed, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in detailed_rows:
        writer.writerow({k: row.get(k, None) for k in fieldnames})

print(f"Saved detailed CSV: {csv_detailed}")


# -----------------------------
# 6) Write full-output CSV (raw printout per run)
# -----------------------------
with open(csv_full, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fps_requested", "rows", "cols", "adc_bits", "full_output"])
    writer.writeheader()
    for row in full_output_rows:
        writer.writerow(row)

print(f"Saved full output CSV (raw text): {csv_full}")


# -----------------------------
# 7) Plot Power vs Resolution
# -----------------------------
if len(plot_x_pixels) >= 2:
    plt.figure()
    plt.plot(plot_x_pixels, plot_power_mW, marker="o")
    plt.xlabel("Resolution (total pixels = rows × cols)")
    plt.ylabel("Total Power (mW)")
    plt.title(f"Power vs Resolution (FPS={fps_fixed}, ADC={adc_bits}-bit)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_name, dpi=200)
    print(f"Saved plot: {png_name}")
else:
    print("Not enough valid points to plot (need >=2).")
