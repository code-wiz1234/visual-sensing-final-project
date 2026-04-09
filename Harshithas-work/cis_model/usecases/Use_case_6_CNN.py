import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from Top_10_22_CNN_optical import CIS_Array
from mpl_toolkits.mplot3d import Axes3D

# Define ranges for rows and columns
num_rows_range = 128  # 15 points from 480 to 2400
num_cols_range = 160  # 15 points from 640 to 1920

# Create meshgrid for 3D plotting
ROWS, COLS = np.meshgrid(num_rows_range, num_cols_range)

# Initialize power array
POWER = np.zeros_like(ROWS)

# Constants from your configuration
PIXEL_BINNING = [1, 1]
INPUT_CLK_FREQ = 5e5  # 100 MHz (note: 1e5 is 100 kHz, you might mean 1e8)
PGA_DC_GAIN = 8
NUM_MUX_INPUT = 0
NUM_SUBFRAMES = 1
INPUT_PIXEL_MAP  = [
    [0.2, 9, 9],
    [0.2, 9, 9],
    [0.2, 9, 9],
    [0.2, 9, 9],
    [0.2, 9, 9]
]

  # Assuming default value

# CIS_Array class instantiation with parameter explanations:
# Based on Top_10_22_CNN_optical.py documentation
sensor = CIS_Array(
    # Array Configuration:
    pixel_binning_map=PIXEL_BINNING,  # Pixel binning mapping [horizontal_binning, vertical_binning]
    num_rows=int(num_rows_range),  # Number of rows in the array
    num_cols=int(num_cols_range),  # Number of columns in the array
    Pixel_type=0,  # Pixel type index: 0=4T_APS, 1=CTIA, 2=CNN_optical_PD(PMW), 3=PMW, 4=3T_APS
    CIS_type=2,  # CIS type: 0=Normal, 1=Coded Exposure, 2=CNN
    CFA_type=0,  # Color filter array type: 0=Bayer, 1=RGBE, 2=RYYB, 3=CYYM, 4=CYGM, 5=RGBW, 6=RGBW#1, 7=No filter
    input_pixel_map=INPUT_PIXEL_MAP,  # 3D matrix [%, length, width] of pixel sizes. Example: [0.2, 1.5, 1.5] means 20% of pixels are 1.5x1.5um
    MUX_type=0,  # Multiplexer type: 0=No MUX, 1=MUX before readout
    num_mux_input=1,  # Number of mux inputs (for structures with mux before readout)
    IO_type=1,  # I/O interface type: 0=None, 1=MIPI
    
    # Photodiode/Physical Parameters:
    photodiode_type=0,  # Type of photodiode: 0=Silicon, 1=Perovskite, 2=Foveon
    pd_E=1000,  # Optical power density in W/m²
    num_PD_per_tap=1,  # Number of photodiodes per tap (1 by default; multi-PD designs use multiple PDs)
    num_of_tap=1,  # Number of used taps (1 by default; multi-tap designs like coded exposure use multiple taps)
    num_of_unused_tap=0,  # Number of unused taps (for multi-tap PD designs with some taps unused)
    feature_size_nm=65,  # Process technology node in nm
    
    # Circuit Configuration:
    ADC_type=0,  # ADC type: 0=SS_ADC, 1=SAR_ADC
    adc_resolution=10,  # ADC resolution in bits
    CDS_type=1,  # CDS type: 0=No CDS, 1=CDS, 2=digital CDS
    CDS_amp_gain=8,  # CDS amplifier gain
    PGA_type=0,  # PGA type: 0=No PGA, 1=PGA, 2=two_stage PGA
    PGA_DC_gain=PGA_DC_GAIN,  # PGA DC gain
    if_PLL=0,  # If PLL is used: 0=No, 1=Yes
    
    # Timing Parameters:
    frame_rate=268,  # Frame rate in fps
    subframe_rates=268,  # Subframe rates in fps (for coded exposure design)
    max_subframe_rates=1,  # Maximum subframe rates in fps (max supported subframe rates for coded exposure)
    exposure_time=0,  # Exposure time in s (0=use default value modeled in system)
    input_clk_freq=INPUT_CLK_FREQ,  # Input clock frequency in Hz (to the PLL)
    PLL_output_frequency=INPUT_CLK_FREQ,  # PLL output frequency in Hz (should be larger than ADC output frequency)
    additional_latency=0,  # Additional latency in s (latency not modeled in the system)
    
    # Voltage/Power Parameters:
    analog_V_dd=1.2,  # Analog supply voltage in V
    digital_V_dd=0.8,  # Digital supply voltage in V
    bias_voltage=0.45,  # Bias voltage in V (used to bias source follower of the pixel)
    comparator_bias_voltage=0.45,  # Comparator bias voltage in V (used to bias comparators in ADC, CDS, PGA, etc.)
    
    # CNN/Coded Exposure Parameters:
    CNN_kernel=3,  # CNN kernel configuration (for CNN design)
)