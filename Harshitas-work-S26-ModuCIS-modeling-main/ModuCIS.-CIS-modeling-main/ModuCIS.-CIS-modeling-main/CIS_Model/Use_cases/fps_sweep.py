# fps_sweep.py
# Run from Use_cases folder:
#   python3 fps_sweep.py

import sys
import os
import csv
import re
import io
from contextlib import redirect_stdout

# Allow imports from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from Top_10_22_CNN_optical import CIS_Array

# -----------------------------
# 1) Sweep settings
# -----------------------------
fps_values = [5, 10, 15, 20, 30]   # edit as needed
rows, cols = 480, 640              # fixed for FPS sweep
adc_bits = 10                      # fixed for FPS sweep

# Output files
csv_detailed = "fps_sweep_detailed.csv"
csv_full = "fps_sweep_full_output.csv"
png_name = "power_vs_fps.png"

# -----------------------------
# 2) Fixed configuration (same style as your Use_case_1)
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

def _find_int(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return int(m.group(1)) if m else None

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
    

    # Power (W or mW in printout; your output shows mW for blocks, but total in mW line too)
    # Total
    # Example: "System Total Power : 864.90 mW"
    total_mw = _find_float(r"System Total Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["system_total_power_mW_printed"] = total_mw
    metrics["system_total_power_W_printed"] = (total_mw / 1000.0) if total_mw is not None else None

    # Block powers (mW)
    metrics["pixel_array_power_mW"] = _find_float(r"Pixel Array Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["readout_circuit_power_mW"] = _find_float(r"Readout Circuit Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["pga_power_mW"] = _find_float(r"PGA Power\s*:\s*([0-9.]+)\s*mW", out)

    # ADC could be "SS ADC Power" or "SAR ADC Power" depending on config
    metrics["ss_adc_power_mW"] = _find_float(r"SS ADC Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["sar_adc_power_mW"] = _find_float(r"SAR ADC Power\s*:\s*([0-9.]+)\s*mW", out)

    metrics["mipi_power_mW"] = _find_float(r"MIPI Power\s*:\s*([0-9.]+)\s*mW", out)
    metrics["pll_power_mW"] = _find_float(r"Phase Locked Loop Power\s*:\s*([0-9.]+)\s*mW", out)

    # Noise / DR
    metrics["dynamic_range_dB_first"] = _find_float(r"Dynamic range\s*\n\s*0th Color,\s*0th pixel size\s*:\s*([0-9.]+)\s*dB", out)
    metrics["total_noise_e"] = _find_float(r"Total Noise\s*:\s*([0-9.]+)\s*e-", out)

    # Quantization noise sometimes in "ue-" etc; store raw value if found
    metrics["adc_quant_noise_ue"] = _find_float(r"ADC Quantization Noise\s*:\s*([0-9.]+)\s*ue-", out)

    # Warning flag if FPS clipped
    metrics["fps_clipped_warning"] = 1 if "Input frame rate is too high" in out else 0

    return metrics

# -----------------------------
# 4) Run sweep + capture outputs
# -----------------------------
detailed_rows = []
full_output_rows = []

plot_fps = []
plot_power_mW = []

print("\n=== FPS Sweep (captures full printed output) ===")
print(f"Fixed: resolution={rows}x{cols}, adc_bits={adc_bits}\n")

for fps in fps_values:
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

            frame_rate=float(fps),
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

        # Also print the object attribute total power (this is what you were printing as "Total Power: 0.136...")
        # Keep it so it is captured too:
        try:
            print(f"Total Power (sensor.system_total_power) : {sensor.system_total_power}")
        except Exception as e:
            print("Could not read sensor.system_total_power:", e)

    out = buf.getvalue()

    metrics = parse_metrics(out)
    metrics["fps_requested"] = fps
    metrics["rows"] = rows
    metrics["cols"] = cols
    metrics["adc_bits"] = adc_bits

    # Grab attribute power too (often equals W)
    # Example: sensor.system_total_power -> appears like 0.136...
    # We'll store it separately as "system_total_power_attr"
    attr_power = None
    try:
        attr_power = float(sensor.system_total_power)
    except Exception:
        attr_power = None
    metrics["system_total_power_attr_W"] = attr_power
    metrics["system_total_power_attr_mW"] = (attr_power * 1000.0) if attr_power is not None else None

    detailed_rows.append(metrics)
    full_output_rows.append({
        "fps_requested": fps,
        "rows": rows,
        "cols": cols,
        "adc_bits": adc_bits,
        "full_output": out
    })

    # For plot, prefer attribute power if available, else printed total power
    p_mW = metrics["system_total_power_attr_mW"]
    if p_mW is None:
        p_mW = metrics["system_total_power_mW_printed"]
    if p_mW is not None:
        plot_fps.append(fps)
        plot_power_mW.append(p_mW)

    # Console summary
    print(f"FPS {fps:>3} | Power ≈ {p_mW:.2f} mW" if p_mW is not None else f"FPS {fps:>3} | Power: N/A")

print("\n✅ Sweep complete.\n")

# -----------------------------
# 5) Write detailed CSV
# -----------------------------
# Choose column order
fieldnames = [
    "fps_requested","rows","cols","adc_bits",
    "fps_clipped_warning",
    "effective_pixels",
    "frame_time_ms","exposure_time_us","readout_time_us","io_time_us","adc_time_us","idle_time_us",
    "frame_rate_hz_printed","max_frame_rate_hz",
    "system_total_power_attr_W","system_total_power_attr_mW",
    "system_total_power_mW_printed","system_total_power_W_printed",
    "pixel_array_power_mW","readout_circuit_power_mW","pga_power_mW",
    "sar_adc_power_mW","ss_adc_power_mW","mipi_power_mW","pll_power_mW",
    "dynamic_range_dB_first","total_noise_e","adc_quant_noise_ue"
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
    writer = csv.DictWriter(f, fieldnames=["fps_requested","rows","cols","adc_bits","full_output"])
    writer.writeheader()
    for row in full_output_rows:
        writer.writerow(row)

print(f"Saved full output CSV (raw text): {csv_full}")

# -----------------------------
# 7) Plot Power vs FPS
# -----------------------------
if len(plot_fps) >= 2:
    plt.figure()
    plt.plot(plot_fps, plot_power_mW, marker="o")
    plt.xlabel("Requested Frame Rate (FPS)")
    plt.ylabel("Total Power (mW)")
    plt.title(f"ModuCIS: Power vs FPS (Resolution {rows}x{cols}, ADC {adc_bits}-bit)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_name, dpi=200)
    print(f"Saved plot: {png_name}")
else:
    print("Not enough valid points to plot (need >=2).")
