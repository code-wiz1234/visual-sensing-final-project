# CIS Model Use Cases

This directory contains example use cases and demonstration scripts for the CIS (CMOS Image Sensor) modeling framework.

## Overview

The CIS Model is a comprehensive framework for modeling and analyzing CMOS Image Sensor arrays, including:
- **Pixel arrays** (4T_APS, 3T_APS, CTIA, PMW)
- **Readout circuits** (PGA, CDS, ADC - SS_ADC and SAR_ADC)
- **Timing analysis** (frame rate, exposure time, readout time)
- **Power consumption analysis** (pixel power, readout power, bus power, etc.)
- **Noise analysis** (shot noise, thermal noise, quantization noise, etc.)
- **Various CIS types** (Normal, Coded Exposure, CNN)

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages:
  - numpy
  - matplotlib

### Running Use Cases

All use case scripts are located in this directory. To run a use case:

```bash
# From the main CIS_Model directory
python Use_cases/Use_case_X.py

# Or from this directory
cd Use_cases
python Use_case_X.py
```

The scripts automatically add the parent directory to the Python path, so all imports will work correctly.

## Use Case Categories

### Use Case 1: Conventional CIS
- **Use_case_1_Conventional_CIS.py**: Conventional CMOS Image Sensor configuration example

### Use Case 2: 3D Structure
- **Use_case_2_3D_Structure.py**: 3D stacked pixel structure analysis (e.g., Foveon, Perovskite)
- Link for Perovskite Paper: https://www.nature.com/articles/s41586-025-09062-3
- Link for the Foveon: https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2006/ICIS-0-736/33720.pdf

### Use Case 3: Coded Exposure
- **Use_case_3_coded_exposure.py**: Coded exposure sensor design example
- Link for related papers: https://ieeexplore.ieee.org/document/9723459 and https://ieeexplore.ieee.org/document/8844261

### Use Case 4: ICCP
- **Use_case_4_ICCP.py**: ICCP example, where they define pixels with non-unifor pixel size
- Link for the Paper: https://arxiv.org/abs/2304.14736

### Use Case 5: Sony
- **Use_case_5_Sony.py**: Sony sensor configuration example
- Link for the reference paper: https://ieeexplore.ieee.org/document/8310196

### Use Case 6: CNN
- **Use_case_6_CNN.py**: CNN-based sensor example
- Link for the reference paper: https://ieeexplore.ieee.org/document/9731675

### Use Case 7: CNN with Optical Processing
- **Use_case_7_CNN_with_optical.py**: CNN sensor with optical processing

## Understanding the CIS_Array Class

All use cases instantiate the `CIS_Array` class from either `Top_10_22_CNN_optical` or `Top_10_14_In_pixel_CNN`. The class takes numerous parameters organized into categories:

### Array Configuration
- `num_rows`, `num_cols`: Array dimensions
- `Pixel_type`: 0=4T_APS, 1=CTIA, 2=CNN_PD(PMW), 3=CNN_PD(PWM with CNN), 4=3T_APS
- `CIS_type`: 0=Normal, 1=Coded Exposure, 2=CNN
- `CFA_type`: 0=Bayer, 1=RGBE, 2=RYYB, 3=CYYM, 4=CYGM, 5=RGBW, 6=RGBW#1, 7=No filter
- `input_pixel_map`: 3D matrix [%, length, width] of pixel sizes
- `pixel_binning_map`: [horizontal_binning, vertical_binning]
- `MUX_type`: 0=No MUX, 1=MUX before readout
- `IO_type`: 0=None, 1=MIPI

### Photodiode/Physical Parameters
- `photodiode_type`: 0=Silicon, 1=Perovskite, 2=Foveon
- `pd_E`: Optical power density in W/m²
- `num_PD_per_tap`: Number of photodiodes per tap
- `num_of_tap`: Number of used taps
- `num_of_unused_tap`: Number of unused taps
- `feature_size_nm`: Process technology node in nm
- `PD_saturation_level`: PD saturation level in V

### Circuit Configuration
- `ADC_type`: 0=SS_ADC, 1=SAR_ADC
- `adc_resolution`: ADC resolution in bits
- `CDS_type`: 0=No CDS, 1=CDS, 2=digital CDS
- `CDS_amp_gain`: CDS amplifier gain
- `PGA_type`: 0=No PGA, 1=PGA, 2=two_stage PGA
- `PGA_DC_gain`: PGA DC gain
- `if_PLL`: 0=No, 1=Yes

### Timing Parameters
- `frame_rate`: Frame rate in fps
- `subframe_rates`: Subframe rates in fps (for coded exposure)
- `max_subframe_rates`: Maximum subframe rates in fps
- `exposure_time`: Exposure time in s (0=use default)
- `input_clk_freq`: Input clock frequency in Hz
- `PLL_output_frequency`: PLL output frequency in Hz
- `ADC_input_clk_freq`: ADC input clock frequency in Hz
- `additional_latency`: Additional latency in s

### Voltage/Power Parameters
- `analog_V_dd`: Analog supply voltage in V
- `digital_V_dd`: Digital supply voltage in V
- `bias_voltage`: Bias voltage in V (for source follower)
- `comparator_bias_voltage`: Comparator bias voltage in V
- `ramp_bias_voltage`: Ramp generator bias voltage in V

### CNN/Coded Exposure Parameters
- `CNN_kernel`: CNN kernel configuration
- `deserializer_output_bits`: Deserializer output bits

## Output Metrics

After instantiating a `CIS_Array` object, you can access various metrics:

### Timing Metrics
- `frame_time`: Frame time in seconds
- `max_frame_rate`: Maximum frame rate in fps
- `exposure_time`: Actual exposure time used
- `rst_time`: Pixel reset time
- `readout_time`: Readout time

### Power Metrics
- `system_total_power`: Total system power in W
- `total_pixel_power`: Pixel array power in W
- `readout_total_power`: Readout circuit power in W
- `bus_power`: Bus power in W
- `pll_power`: PLL power in W
- `input_driver_total_power`: Input driver power in W

### Noise Metrics
- `total_noise_square`: Total noise variance in electrons²
- `SNR`: Signal-to-noise ratio array (by color and pixel size) in dB
- `DR`: Dynamic range array (by color and pixel size) in dB
- `FWC`: Full well capacity array (by color and pixel size) in electrons

### Example

```python
from Top_10_22_CNN_optical import CIS_Array

# Create a sensor instance
sensor = CIS_Array(
    num_rows=1080,
    num_cols=1920,
    Pixel_type=0,  # 4T_APS
    CIS_type=0,  # Normal CIS
    CFA_type=0,  # Bayer
    # ... other parameters
)

# Access results
print(f"Total Power: {sensor.system_total_power * 1000:.2f} mW")
print(f"SNR: {sensor.SNR[0][0]:.2f} dB")  # SNR for first color, first pixel size
print(f"Frame Rate: {sensor.frame_rate:.2f} fps")
```

## Customizing Use Cases

To create your own use case:

1. Copy an existing use case file as a template
2. Modify the `CIS_Array` parameters to match your design
3. Adjust the analysis or plotting code as needed
4. Run the script

All use cases include detailed inline comments explaining each parameter based on the documentation in `Top_10_22_CNN_optical.py`.

## Notes

- Most use cases generate plots or analysis outputs
- Some use cases perform parameter sweeps (e.g., frame rate, subframe rate)
- Power values are typically in Watts; multiply by 1000 for mW
- Noise values are in electrons or electrons²
- Timing values are in seconds; multiply by appropriate factors for ms/µs

## Reference

For detailed documentation of the CIS_Array class and its parameters, refer to:
- `Top_10_22_CNN_optical.py`: Main CIS_Array implementation with comprehensive documentation

