import numpy as np
from APS import APS, CTIA, CNN_PD
from TIA import TIA
from digital_gate import NOR, NAND, INV
from wire import Wire
from parameter_class import NMOS
from Ramp_generator import Ramp_generator
from Comparator import Comparator
from Counter import Countner
from PLL import Phase_lock_loop
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from Internal_buffer import Repeater
from Input_Driver import Input_Driver
from analog_buffer import Analog_Buffer, Analog_Buffer_Bus
from MIPI import MIPI
from SRAM_Buffer import SRAM, MUX
from Noise import Shot_Noise, Dark_Current_Noise, Thermal_Noise, RTS_Noise, Transfer_Noise, Current_Shot_Noise, ADC_Quantizing_Noise, System_Noise, Reset_Noise
from SS_ADC import SS_ADC
from SAR_ADC import SAR_ADC
from CDS import CDS
from PGA import PGA
from SH import SHCircuit
from In_Pixel_Latch import latch
from In_Pixel_Ctrl import ctrl
from Row_logic import Row_logic
from Deserializer import Deserializer
from Timing_Generator import time_generator
import sys

#Rolling shutter, global shutter, Stacked CIS, per pixel ADC., for the related cap, make it realted to the tech node. documentaions of the relation.
# What considered, what not considered
# CIS (CMOS Image Sensor) Array class
# Complete CIS array model with CNN processing capabilities and optical components
# 
# Inputs:
#   Array Configuration:
#     - num_rows (number of rows), num_cols (number of columns)
#     - Pixel type index: 0=4T_APS, 1=CTIA, 2=CNN_optical_PD(PMW), 3=PMW, 4=3T_APS
#     - CIS_type (CIS type: 0=Normal, 1=Coded Exposure, 2=CNN)
#     - CFA_type (color filter array type: 0=Bayer, 1=RGBE, 2=RYYB, 3=CYYM, 4=CYGM, 5=RGBW, 6=RGBW#1, 7=No filter)
#     - input_pixel_map (input pixel mapping: 3D matrix [%, length, width] of pixel sizes)
#     - pixel_binning_map (pixel binning mapping: [horizontal_binning, vertical_binning])
#   
#   Photodiode/Physical Parameters:
#     - photodiode_type (0=normal, 1=perovskite, 2=Foveon)
#     - pd_E (optical power density in W/m²)
#     - num_PD_per_tap (number of photodiodes per tap)
#     - num_of_tap (number of used taps), num_of_unused_tap (number of unused taps)
#     - PD_saturation_level (PD saturation level in V)
#     - PD_tech (PD technology index, optional)
#     - feature_size_nm (process technology node in nm)
#     - capm2 (capacitance per unit area in F/m², optional), EQE (external quantum efficiency, optional), lambda_m (wavelength in m, optional)
#   
#   Circuit Configuration:
#     - ADC_type (ADC type: 0=SS_ADC, 1=SAR_ADC)
#     - adc_resolution (ADC resolution in bits)
#     - CDS_type (CDS type: 0=No CDS, 1=CDS, 2=digital CDS)
#     - CDS_amp_gain (CDS amplifier gain)
#     - PGA_type (PGA type: 0=No PGA, 1=PGA, 2=two_stage PGA)
#     - PGA_DC_gain (PGA DC gain)
#     - MUX_type (multiplexer type: 0=No MUX, 1=MUX)
#     - num_mux_input (number of mux inputs)
#     - IO_type (I/O interface type: 0=None, 1=MIPI)
#   
#   Timing Parameters:
#     - frame_rate (frame rate in fps)
#     - subframe_rates (subframe rates in fps, for coded exposure)
#     - max_subframe_rates (maximum subframe rates in fps)
#     - exposure_time (exposure time in s, 0=use default)
#     - input_clk_freq (input clock frequency in Hz)
#     - PLL_output_frequency (PLL output frequency in Hz)
#     - ADC_input_clk_freq (ADC input clock frequency in Hz, 0=use input_clk_freq)
#     - additional_latency (additional latency in s)
#     - if_time_generator (if time generator is used: 0=No, 1=Yes)
#   
#   Voltage/Power Parameters:
#     - analog_V_dd (analog supply voltage in V)
#     - digital_V_dd (digital supply voltage in V)
#     - bias_voltage (bias voltage in V, for source follower)
#     - comparator_bias_voltage (comparator bias voltage in V)
#     - ramp_bias_voltage (ramp generator bias voltage in V)
#     - V_ctrl (control voltage in V, for PLL VCO)
#     - if_PLL (if PLL is used: 0=No, 1=Yes)
#   
#   Capacitance Parameters:
#     - load_cap (load capacitance in F)
#     - ramp_load_cap (ramp generator load capacitance in F)
#     - comparator_load_cap (comparator load capacitance in F)
#     - comparator_input_cap (comparator input capacitance in F)
#     - PGA_input_cap (PGA input capacitance in F)
#     - charge_pump_load_cap (charge pump load capacitance in F)
#     - CTIA_C_FB (CTIA feedback capacitance in F)
#     - CTIA_load_cap (CTIA load capacitance in F)
#     - MIPI_unit_cap (MIPI unit capacitance in F)
#   
#   CNN/Coded Exposure Parameters:
#     - CNN_kernel (CNN kernel configuration)
#     - deserializer_output_bits (deserializer output bits)
#   
#   Other:
#     - print_output (if print output is used, optional, default=True)
# 
# Outputs:
#   Timing:
#     - frame_time (frame time in s), max_frame_rate (maximum frame rate in fps)
#   Power:
#     - total_power (total system power in W), pixel_power (pixel array power in W)
#     - adc_power (ADC power in W), pll_power (PLL power in W)
#     - bus_power (bus power in W), input_driver_power (input driver power in W)
#     - MIPI_power (MIPI power in W), SRAM_buffer_power (SRAM buffer power in W)
#   Noise:
#     - total_noise_square (total noise variance in electrons²)
#     - SNR (signal-to-noise ratio in dB), DR (dynamic range in dB)
#     - and various timing and noise metrics
class CIS_Array:
    """
    CIS (CMOS Image Sensor) Array Model
    
    This class models a complete CIS array including:
    - Pixel array (APS, CTIA, PMW)
    - Readout circuits (PGA, CDS, ADC)
    - Timing analysis
    - Power consumption analysis
    - Noise analysis (shot, thermal, quantization, etc.)
    - Transistor count
    
    The __init__ method is organized into logical sections:
    1. Store basic parameters
    2. Configure Color Filter Array
    3. Initialize bus models
    4. Initialize pixel array
    5. Initialize readout circuits
    6. Initialize control circuits
    7. Calculate timing
    8. Calculate power consumption
    9. Calculate noise
    10. Calculate transistor count
    """
    
    # CFA (Color Filter Array) color mappings, Cuurently, the we conly support CFA 0, 1, 5, 6, 7
    CFA_COLORS = {
        0: [0,1,2,1],  # Bayer
        1: [0,1,2,1],  # RGBE
        2: [0,2,5,5],  # RYYB
        3: [5,6,7,5],  # CYYM
        4: [2,5,6,7],  # CYGM
        5: [0,1,2,7],  # RGBW
        6: [0,1,2,8],  # RGBW #1
        7: [8,8,8,8]   # No color filter
    }
    
    def __init__(self, photodiode_type, num_rows, num_cols, pd_E, Pixel_type, MUX_type, IO_type,
                 input_clk_freq, PLL_output_frequency, num_PD_per_tap, PGA_type, CDS_type, CIS_type,
                 CFA_type, input_pixel_map, pixel_binning_map, ADC_type, CDS_amp_gain, additional_latency,
                 num_of_unused_tap, analog_V_dd, digital_V_dd, adc_resolution, PGA_DC_gain,
                 feature_size_nm, frame_rate, subframe_rates, max_subframe_rates, CNN_kernel, exposure_time,
                 bias_voltage=0.6, load_cap=100e-15, ramp_bias_voltage=0.6, ramp_load_cap=100e-15,
                 comparator_bias_voltage=0.6, comparator_load_cap=100e-15, PGA_input_cap=100e-15,
                 comparator_input_cap=100e-15, charge_pump_load_cap=100e-15, V_ctrl=0.6, ADC_input_clk_freq=0,
                 MIPI_unit_cap=0, if_time_generator=0, PD_saturation_level=0.6, PD_tech=0, num_of_tap=2,
                 if_PLL=1, num_mux_input=48, deserializer_output_bits=16, CTIA_C_FB=100e-15, fill_factor=0.5,
                 CTIA_load_cap=100e-15, shutter=0, capm2=None, EQE=None, lambda_m=None, print_output=True):
        """
        Initialize CIS Array
        
        Args:
            Array Configuration:
                num_rows, num_cols: Array dimensions
                Pixel type index: 0=4T_APS, 1=CTIA, 2=CNN_optical_PD(PMW), 3=PMW, 4=3T_APS
                CIS_type: CIS type (0=Normal, 1=Coded Exposure, 2=CNN)
                CFA_type: Color filter array type (0=Bayer, 1=RGBE, 2=RYYB, 3=CYYM, 4=CYGM, 5=RGBW, 6=RGBW#1, 7=No filter)
                input_pixel_map: 3D matrix [%, length, width] of PD sizes. Example: [0.2, 1.5, 1.5] means 20% of pixels are 1.5x1.5um
                pixel_binning_map: Pixel binning mapping [horizontal_binning, vertical_binning]
                MUX_type: Multiplexer type (0=No MUX, 1=MUX before readout)
                num_mux_input: Number of mux inputs (for structures with mux before readout)
                IO_type: I/O interface type (0=None, 1=MIPI)
            
            Photodiode/Physical Parameters:
                photodiode_type: Type of photodiode (0=Silicon, 1=Perovskite, 2=Foveon)
                pd_E: Optical power density in W/m²
                num_PD_per_tap: Number of photodiodes per tap (1 by default; multi-PD designs use multiple PDs (Sony's design))
                num_of_tap: Number of used taps (1 by default; multi-tap designs like coded exposure use multiple taps)
                num_of_unused_tap: Number of unused taps (for multi-tap PD designs with some taps unused)
                PD_saturation_level: PD saturation level in V (voltage level when PD is saturated, related to PD technology)
                PD_tech: PD technology index (not currently used, kept for future reference; can be used in photodiode.py)
                feature_size_nm: Process technology node in nm
                capm2: Capacitance per unit area in F/m² (optional, default from photodiode.py; used for PD property)
                EQE: External quantum efficiency (optional, default from photodiode.py; used for PD property)
                lambda_m: Wavelength in m (optional, default from photodiode.py; used for PD property)
                fill_factor: Fill factor of the pixel; used for the effective EQE calculation
            
            Circuit Configuration:
                ADC_type: ADC type (0=SS_ADC, 1=SAR_ADC)
                adc_resolution: ADC resolution in bits
                CDS_type: CDS type (0=No CDS, 1=CDS, 2=digital CDS)
                CDS_amp_gain: CDS amplifier gain
                PGA_type: PGA type (0=No PGA, 1=PGA, 2=two_stage PGA)
                PGA_DC_gain: PGA DC gain
                if_PLL: If PLL is used (0=No, 1=Yes)
            
            Timing Parameters:
                frame_rate: Frame rate in fps
                subframe_rates: Subframe rates in fps (for coded exposure design)
                max_subframe_rates: Maximum subframe rates in fps (max supported subframe rates for coded exposure)
                exposure_time: Exposure time in s (0=use default value modeled in system)
                input_clk_freq: Input clock frequency in Hz (to the PLL)
                PLL_output_frequency: PLL output frequency in Hz (should be larger than ADC output frequency)
                ADC_input_clk_freq: ADC input clock frequency in Hz (0=use input_clk_freq; clock frequency of ADC and PLL output)
                additional_latency: Additional latency in s (latency not modeled in the system)
                if_time_generator: If time generator is used (0=No, 1=Yes; for coded exposure design)
            
            Voltage/Power Parameters:
                analog_V_dd: Analog supply voltage in V
                digital_V_dd: Digital supply voltage in V
                bias_voltage: Bias voltage in V (used to bias source follower of the pixel)
                comparator_bias_voltage: Comparator bias voltage in V (used to bias comparators in ADC, CDS, PGA, etc.)
                ramp_bias_voltage: Ramp generator bias voltage in V
                V_ctrl: Control voltage in V (for voltage controlled oscillator of PLL)
            
            Capacitance Parameters:
                load_cap: Load capacitance in F (connected to output of pixel; directly connected to Source Follower)
                ramp_load_cap: Ramp generator load capacitance in F
                comparator_load_cap: Comparator load capacitance in F
                comparator_input_cap: Comparator input capacitance in F
                PGA_input_cap: PGA input capacitance in F (directly connected to Source Follower of pixel)
                charge_pump_load_cap: Charge pump load capacitance in F (key part of PLL)
                CTIA_C_FB: CTIA feedback capacitance in F (used for CTIA pixel type)
                CTIA_load_cap: CTIA load capacitance in F (used for CTIA pixel type)
                MIPI_unit_cap: MIPI unit capacitance in F
            
            CNN/Coded Exposure Parameters:
                CNN_kernel: CNN kernel configuration (for CNN design)
                deserializer_output_bits: Deserializer output bits (mainly used for coded exposure design)
            
            Other:
                print_output: If print output is used (optional, default=True; for debug purpose)
        """
        # ==================== SECTION 1: Store Basic Parameters ====================
        self.print_output = print_output
        self.ADC_input_freq = ADC_input_clk_freq if ADC_input_clk_freq != 0 else input_clk_freq
        self.subframe_rates = subframe_rates
        self.analog_V_dd = analog_V_dd
        self.digital_V_dd = digital_V_dd
        self.load_cap = load_cap
        self.output_bus_length = num_rows * 1e-6 #in meter
        self.bias_voltage = bias_voltage
        self.deserializer_output_bits = deserializer_output_bits
        self.photodiode_type = photodiode_type
        self.pixel_binning_map = pixel_binning_map

        # Check if the number of PD per tap is supported for the pixel type. Now, only 4T_APS and 3T_APS support multiple PD design in the system.
        if num_PD_per_tap != 1 and Pixel_type != 0:
            sys.exit("only normal pixel type support multiple PD design")

        # ==================== SECTION 2: Configure Color Filter Array ====================
        # Configure the effective pixels, 3D stacked design each pixel is treated as an effective pixel, as it has three layers of PDs and each layer capture one color.
        if photodiode_type == 1 or photodiode_type == 2:
            self.effective_pixels = num_cols * num_rows
            num_cols = num_cols * 3
            self.color = [0, 1, 2, 0]
        # For normal color filter array design, 4 pixles are treated as an effective pixel (RGBW, RGBW#1, RYYB, CYYM, CYGM)
        # For monochrome design, each pixel is treated as an effective pixel. (CFA type == 7)
        elif photodiode_type == 0:
            if CFA_type in self.CFA_COLORS:
                self.color = self.CFA_COLORS[CFA_type]
                self.effective_pixels = num_cols * num_rows / 4 if CFA_type != 7 else num_cols * num_rows
            else:
                sys.exit("Error: no such color filter")
        else:
            sys.exit("Error: no such photodiode")

        # ==================== SECTION 3: Initialize Basic Components ====================
        self.inv = INV(
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd
        )

        self.bias_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.nor = NOR(
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd
        )

        self.nand = NAND(
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd
        )

        # Initialize the Transimpedance Amplifier, which is used for the PMW design.
        self.TIA = TIA(
            bias_voltage=bias_voltage,
            input_clk_freq=frame_rate,
            input_cap=100e-15,
            DC_gain=2000,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            num_cols=num_cols/CNN_kernel
        )

        # Compute the bias current for the bias transistor
        self.bias_current = self.compute_bias_current()

        # ==================== SECTION 4: Initialize Bus Models ====================
        # Two types of bus are supported: global_aggressive and semi_aggressive.
        if num_rows > 100:
            self.bus_wire_type = "global_aggressive"
        else:
            self.bus_wire_type = "semi_aggressive"

        # estimate the bus length
        self.horizontal_bus_length = num_cols * input_pixel_map[0][0]*1e-6
        self.vertical_bus_length = num_rows * input_pixel_map[0][1]*1e-6

        self.input_bus = Wire(
            wire_type=self.bus_wire_type,
            feature_size_nm=feature_size_nm,
            temperature=293
        )

        self.output_bus = Wire(
            wire_type=self.bus_wire_type,
            feature_size_nm=feature_size_nm,
            temperature=293
        )

        # Initialize the Analog Buffer Bus, the bus used to transfer the analog signal. Such as the input but to ramp generator.
        self.Analog_Buffer = Analog_Buffer_Bus(
            wire_unit_cap=self.input_bus.cap_wire_per_m,
            wire_unit_res=self.input_bus.res_wire_per_m,
            feature_size_nm=feature_size_nm,
            V_dd=self.analog_V_dd,
            bias_voltage=self.bias_voltage
        )
        
        #each repeater is used to buffer the long bus
        self.buffer = Repeater(
            feature_size_nm=feature_size_nm,
            capWirePerUnit=self.input_bus.cap_wire_per_m,
            resWirePerUnit=self.input_bus.res_wire_per_m,
            V_dd=self.analog_V_dd
        )

        
        # ==================== SECTION 5: Initialize Pixel Array ====================
        # Initialize the pixel array, the pixel array is the core of the system.

        # Initialize the swing voltage of the PD, which is the voltage level when the PD is saturated.
        V_swing = PD_saturation_level
        self.pixels = []
        
        # Common pixel parameters
        common_params = {
            'num_PD_per_tap': num_PD_per_tap,
            'photodiode_type': self.photodiode_type,
            'pd_E': pd_E,
            'feature_size_nm': feature_size_nm,
            'bias_current': self.bias_current,
            'V_dd': analog_V_dd,
            'V_swing': V_swing,
            'exposure_time': exposure_time,
            'PD_tech': PD_tech,
            'capm2': capm2,
            'EQE': EQE ,
            'lambda_m': lambda_m
        }
        
        # Initialize the pixel array, the pixel array is the core of the system.
        # For the normal pixel type, 4T_APS and 3T_APS are supported.
        # For the coded exposure design, APS and CTIA are supported.
        # For the CNN design, CNN_PD(PMW) is supported.
        for i in range(4):
            temp_pixel_set = []
            for j in range(5):
                pixel_params = {
                    'color': self.color[i],
                    'pd_length': input_pixel_map[j][1] * math.sqrt(fill_factor),
                    'pd_width': input_pixel_map[j][2] * math.sqrt(fill_factor),
                    **common_params
                }
                
                if Pixel_type == 0 or Pixel_type == 4:
                    pixel_params.update({'num_of_tap': num_of_tap, 'Pixel_type': Pixel_type})
                    temp_pixel_instance = APS(**pixel_params)
                elif Pixel_type == 1:
                    pixel_params.update({
                        'num_of_tap': num_of_tap + num_of_unused_tap,
                        'Cap_FB': CTIA_C_FB,
                        'load_cap': CTIA_load_cap
                    })
                    temp_pixel_instance = CTIA(**pixel_params)
                elif Pixel_type == 2 or 3:
                    temp_pixel_instance = CNN_PD(**pixel_params)
                else:
                    sys.exit("NO Such Pixel type")
                
                temp_pixel_set.append(temp_pixel_instance)
            self.pixels.append(temp_pixel_set)
        # ==================== SECTION 6: Initialize Readout Circuits ====================

        # Initialize the MUX, the MUX is used to select the input signal to the ADC.
        self.mux = MUX(
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            num_input=num_mux_input,
            next_stage_input_Cap=0
        )

        # Initialize the PGA, the PGA is used to amplify the signal.
        self.PGA = PGA(
            bias_voltage = bias_voltage, 
            input_clk_freq = frame_rate, #Update later
            input_cap = PGA_input_cap, 
            DC_gain = PGA_DC_gain, 
            feature_size_nm = feature_size_nm, 
            V_dd=analog_V_dd,
            frame_rate = frame_rate,
            num_cols = num_cols,
            num_rows = num_rows
        )
    
        # Initialize the Comparator, this comparator is mainly used for the CNN sensor readout design.
        self.comparator = Comparator(
            bias_voltage=comparator_bias_voltage,
            load_cap=comparator_load_cap,
            input_cap=comparator_input_cap,
            DC_gain = 2000,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd
        )

        # Calculate output voltage range
        self.min_output_voltage = self.bias_voltage - self.bias_transistor.v_th
        self.max_output_voltage = self.compute_sf_out_voltage(self.pixels[0][0].rst_voltage)

        # Initialize the CDS, the CDS is used to remvoe the fixed pattern noise.
        self.CDS = CDS (V_rst = self.max_output_voltage, V_sig = self.min_output_voltage, comaprator_FB_cap = CDS_amp_gain*comparator_input_cap,
            comparator_bias_voltage = comparator_bias_voltage, comparator_load_cap = comparator_load_cap, comparator_input_cap = comparator_input_cap, 
            num_cols = num_cols, num_rows = num_rows, frame_rate = frame_rate,
            feature_size_nm = feature_size_nm, V_dd=analog_V_dd, DC_gain = 2000
        )
        
        self.SAR_ADC_input_clk_freq = self.ADC_input_freq

        # Initialize the SAR ADC, the SAR ADC is used to convert the analog signal to the digital signal.
        self.SAR_ADC = SAR_ADC(
            max_output_voltage = self.CDS.CDS_output_voltage, adc_resolution = adc_resolution,
            comparator_bias_voltage = comparator_bias_voltage, comparator_load_cap = comparator_load_cap, comparator_input_cap = comparator_input_cap, 
            num_cols = num_cols, num_rows = num_rows, frame_rate = frame_rate,
            input_clk_freq = self.SAR_ADC_input_clk_freq, feature_size_nm = feature_size_nm, V_dd=digital_V_dd, DC_gain = 2000
        )

        # Initialize the SS ADC, the SS ADC is used to convert the analog signal to the digital signal.
        self.SS_ADC = SS_ADC(
            ramp_bias_voltage = ramp_bias_voltage, ramp_load_cap = ramp_load_cap, input_clk_freq=self.ADC_input_freq, 
            max_output_voltage = self.digital_V_dd, adc_resolution = adc_resolution,
            comparator_bias_voltage = comparator_bias_voltage, comparator_load_cap = comparator_load_cap, comparator_input_cap = comparator_input_cap, 
            num_cols = num_cols, num_rows = num_rows, feature_size_nm = feature_size_nm, V_dd=digital_V_dd, frame_rate=frame_rate
        )

        #The followiug three components are only used for the CNN sensor readout design. To clear say how do they work, please refer to the paper from National Tsing Hua University.
        # Initialize the PMW Comparator, the PMW Comparator is the core of the PMW design.
        self.PMW_comparator = Comparator(
            bias_voltage=comparator_bias_voltage,
            load_cap=comparator_load_cap,
            input_cap=comparator_input_cap,
            DC_gain = 2000,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd
        )
        
        # Initialize the PMW Ramp Generator, the PMW Ramp Generator is used to generate the pulse width modulated signal.
        self.PMW_ramp_generator = Ramp_generator(
            bias_voltage=ramp_bias_voltage,
            load_cap=ramp_load_cap,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            input_clk_freq=input_clk_freq,
            adc_resolution=4
        )

        # Initialize the Readout Ramp Generator, it is used in the CNN readout design.
        self.readout_ramp_generator = Ramp_generator(
            bias_voltage=ramp_bias_voltage,
            load_cap=ramp_load_cap,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            input_clk_freq=input_clk_freq,
            adc_resolution=4
        )

        # Initialize the PLL, the PLL is used to generate the clock signal for the system.
        # PLL Low-Pass Filter (divider is the resolution of the LPF difference)
        self.LPF_resistance, self.LPF_load_cap = self.compute_rc_lpf(input_clk_freq, divider=10)
        if ADC_type == 1:
            self.PLL_output_frequency = self.SS_ADC.counter_clk_freq
        else:
            self.PLL_output_frequency = PLL_output_frequency

        self.PLL = Phase_lock_loop(
            CP_load_cap = charge_pump_load_cap, 
            LPF_resistance = self.LPF_resistance,
            LPF_load_cap = self.LPF_load_cap, 
            input_clk_freq = input_clk_freq, 
            output_clk_frequency = self.PLL_output_frequency, 
            bias_votlage = self.bias_voltage, 
            feature_size_nm = feature_size_nm, 
            V_dd=self.digital_V_dd,
            V_ctrl=V_ctrl
        )

        # Initialize the SH Circuit, the SH Circuit is used to store the signal.
        self.SH = SHCircuit( 
            V_dd = analog_V_dd,
            load_cap = load_cap, 
            feature_size_nm = feature_size_nm
        )

        # The following two counters are only used for the CNN sensor readout design. To clear say how do they work, please refer to the paper from National Tsing Hua University.
        # Initialize the CNN Case Counter
        self.CNN_case_counter = Countner(
            ADC_resolution = 4,
            input_clk_freq = num_cols*num_rows/(4*(CNN_kernel**2)) * frame_rate*4,
            feature_size_nm = feature_size_nm, 
            V_dd=digital_V_dd
        )

        # Initialize the Up Down Counter
        self.up_down_counter = Countner(
            ADC_resolution = 15,
            input_clk_freq = num_cols*num_rows/(4*(CNN_kernel**2)) * 4,
            feature_size_nm = feature_size_nm, 
            V_dd=digital_V_dd
        )
        
        # ==================== SECTION 6: Initialize Control Circuits ====================
        # Initialize the Output Buffer, it is used to buffer the output signal.
        self.output_buffer = INV(feature_size_nm=feature_size_nm, V_dd=digital_V_dd)
        
        #The following ctrl logics are mainly used for the coded exposure design. For reference, please refer to the paper from Toroto university, or other coded exposure design papers.
        # Initialize the Deserializer, it is used to deserialize the input control signals, especially for the coded exposure design.
        self.deserializer = Deserializer(
            input_clk_freq=input_clk_freq,
            feature_size_nm=feature_size_nm,
            V_dd=digital_V_dd,
            output_bits=self.deserializer_output_bits
        )
        
        # Initialize the Row Logic, it is used to generate the row control signals. Mainly used for the coded exposure design.
        self.row_logic = Row_logic(
            input_clk_freq=input_clk_freq,
            feature_size_nm=feature_size_nm,
            V_dd=digital_V_dd
        )
        
        # Initialize the Time Generator, it is used to generate the time control signals. Mainly used for the coded exposure design.
        self.time_generator = time_generator(
            input_clk_freq=input_clk_freq,
            feature_size_nm=feature_size_nm,
            V_dd=digital_V_dd,
            denominator=self.deserializer_output_bits
        )
        
        # Initialize the Subframe Control Input Driver, it is used to drive the subframe control signals.
        self.Subframe_ctrl_input_driver = Input_Driver(
            input_clk_freq=subframe_rates,
            num_rows=num_rows,
            feature_size_nm=feature_size_nm,
            V_dd=self.digital_V_dd
        )
        
        # Initialize the Frame Control Input Driver, it is used to drive the frame control signals.
        self.frame_ctrl_input_driver = Input_Driver(
            input_clk_freq=frame_rate,
            num_rows=num_rows,
            feature_size_nm=feature_size_nm,
            V_dd=self.digital_V_dd
        )
        
        # Initialize the In Pixel Latch, it is used to latch the input pixel signals.
        self.num_bit = num_of_tap
        self.in_pixel_latch = latch(
            clk_freq=subframe_rates*frame_rate,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            num_bit=self.num_bit
        )
        
        # Initialize the In Pixel Control, it is used to control the input pixel signals.
        self.in_pixel_ctrl = ctrl(
            clk_freq=subframe_rates*frame_rate,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            num_bit=self.num_bit
        )
        
        # Initialize the MIPI, it is used to interface the sensor with the external world.
        self.MIPI = MIPI(
            data_bandwidth=adc_resolution*num_cols*num_rows/(pixel_binning_map[0]*pixel_binning_map[1]),
            frame_rate=frame_rate,
            MIPI_unit_cap=MIPI_unit_cap,
            feature_size_nm=feature_size_nm,
            MIPI_unit_energy=100e-12
        )
        
        # ==================== SECTION 7: Calculate Timing ====================
        self.additional_latency = additional_latency

        # Calculate the reset time of the pixel, the final reset time shown to useris the longest reset time of all the pixels.
        pixel_rst_time = []
        # Loop through all the pixels to get their reset time.
        for i in range(4):
            for j in range(5):
                pixel_rst_time.append(self.pixels[i][j].rst_time)

        self.rst_time = max(pixel_rst_time)
        
        # Calculate readout and floating diffusion (FD) reset times
        # The readout time is the time required to fill the output cap to the max output voltage.
        self.readout_time = self.compute_readout_time(self.max_output_voltage, self.min_output_voltage)
        self.fd_reset_time = max(self.pixels[0][0].fd_rst_time, self.readout_time) if Pixel_type == 0 else 0

        # Calculate exposure time constraints
        self.sf_max_input = self.pixels[0][0].rst_voltage
        self.sf_min_input = self.compute_sf_input_voltage(self.min_output_voltage)
        # The max subframe rate is the maximum subframe rate that can be supported by the system. If max_subframe_rates is not provided, use the model calculated value.
        if max_subframe_rates != 0:
            self.max_subframe_rates = (input_clk_freq/self.deserializer_output_bits)
        else:
            self.max_subframe_rates = max_subframe_rates
        
        # check if the subframe rate is too high, if so, use the model replace it with the max subframe rate.
        if subframe_rates > 1:
            if subframe_rates > self.max_subframe_rates:
                if self.print_output:
                    print("Input sunframe rate is too high, will use largest frame rate calcualted by the model")
                self.subframe_rates = self.max_subframe_rates
            else:
                self.subframe_rates = subframe_rates
        

        # Loop through all the pixels to get their exposure time.
        pixels_exposure_time = []
        for i in range(4):
            for j in range(5):
                temp = self.pixels[i][j].compute_exposure_time(self.pixels[i][j].rst_voltage - self.sf_min_input)
                # For the coded exposure design, the exposure time is based on the subframe rate.
                if CIS_type == 1:
                    temp = temp*self.subframe_rates/self.max_subframe_rates
                pixels_exposure_time.append(temp)
        self.max_exposure_time = min(pixels_exposure_time)

        #if the user provided the exposure time, use it, otherwise, use the model calculated value.
        # If the provided exposure time is larger than max_exposure_time, cap it at max_exposure_time
        self.exposure_time = self.max_exposure_time if exposure_time == 0 else min(exposure_time, self.max_exposure_time)
        
        # Calculate base frame time, which is the time required to complete one frame.
        base_frame_time = self.exposure_time + self.rst_time + self.readout_time
        self.frame_time = base_frame_time + (self.fd_reset_time if Pixel_type == 0 else 0)
        
        # MIPI delay (Reference: AMD https://docs.amd.com/r/en-US/pg232-mipi-csi2-rx/D-PHY-Latency)
        self.MIPI_delay = (49 if adc_resolution <= 8 else 60) / input_clk_freq
        
        # Calculate the selection time, which is the time required to for the analog parts of the readout circuit to finish their work.
        if ADC_type == 0:
            # If no on-chip ADC is used, the selection time is 0.
            self.sel_time = 0
        elif ADC_type == 1:
            if CDS_type == 2:
                # For the SS ADC, it allows both digital and analog CDS.
                self.sel_time = self.SS_ADC.comparator.AZ_time + self.SS_ADC.ramp_generator.CDS_time + self.SS_ADC.ramp_generator.ADC_time + self.SS_ADC.ramp_generator.reset_time
                self.counter_time = self.SS_ADC.ramp_generator.CDS_time + self.SS_ADC.ramp_generator.ADC_time
            else:
                self.sel_time = self.SS_ADC.comparator.AZ_time + self.SS_ADC.ramp_generator.ADC_time + self.SS_ADC.ramp_generator.reset_time
                self.counter_time = self.SS_ADC.ramp_generator.ADC_time
        elif ADC_type == 2:
            self.sel_time = self.SAR_ADC.delay
            self.counter_time = self.SAR_ADC.delay
        else:
            sys.exit("Error: no such ADC")
        # Add component delays to frame time
        self.frame_time += self.sel_time
        
        # Add the PGA delay to the frame time.
        if PGA_type == 1:
            self.frame_time += self.PGA.latency
        elif PGA_type == 2:
            self.frame_time += self.PGA.latency * 2

        # Add the CDS delay to the frame time.
        if CDS_type == 1:
            if (CIS_type != 1):
                self.frame_time += self.CDS.CDS_delay
            # For the coded exposure design, the CDS delay is the time required to finish the SH operation. It is like a CDS for dual type design
            elif CIS_type == 1:
                self.frame_time += self.SH.delay
        elif CDS_type != 0:
            sys.exit("Error: no such CDS")
        
        # Calculate total readout time (excludes exposure time and reset time)
        self.total_readout_time = self.frame_time - self.exposure_time - self.rst_time

        # Add I/O delay
        if IO_type == 1:
            self.frame_time += self.MIPI_delay
        elif IO_type != 0:
            sys.exit("Error: no such IO")
        
        # Calculate maximum frame rate
        # In most cases, exposure time is hidden after total readout time
        # However, if exposure time is longer than total readout time,
        # the time to finish readout of one frame equals frame time
        if shutter == 0:
            if (self.total_readout_time * num_rows/pixel_binning_map[0] < self.exposure_time or ADC_type == 0):
                self.max_frame_rate = 1/self.frame_time
            else:
                self.max_frame_rate = 1/(self.sel_time * (num_rows/pixel_binning_map[0]))
        elif shutter == 1:
            if (self.total_readout_time * num_rows/pixel_binning_map[0] < self.exposure_time or ADC_type == 0):
                #For the global shutter mode, the pixels must wait all pixels finsih readout before next exposure
                self.frame_time += self.sel_time * ((num_rows/pixel_binning_map[0]) - 1)
                self.max_frame_rate = 1/self.frame_time
            else:
                #for the global shutter mode, the exposure and integration time  (base frame time here) cannot be hidden
                self.max_frame_rate = 1/(self.sel_time * (num_rows/pixel_binning_map[0]) + base_frame_time)
        else:
            sys.exit("Error: no such shutter mode")
        
        # Set actual frame rate, if the user provided the frame rate, use it, otherwise, use the model calculated value.
        # if the provided frame rate is larger than max_frame_rate, cap it at max_frame_rate
        self.frame_rate = self.max_frame_rate
        if frame_rate > 1:
            if frame_rate > self.max_frame_rate:
                if self.print_output:
                    print("Input frame rate is too high, will use largest frame rate calculated by the model")
                self.frame_rate = self.max_frame_rate
            else:
                self.frame_rate = frame_rate
        
        self.subframe_time = 1/subframe_rates
        self.subframe_rates_per_second = self.subframe_rates * self.frame_rate
        self.idle_time = 1/self.frame_rate - 1/self.max_frame_rate

        # CNN-specific timing, to understand how it works, please refer to the paper from National Tsing Hua University.
        if CIS_type == 2:
            if Pixel_type == 2:
                self.pixel_time = self.pixels[0][0].compute_exposure_time(self.pixels[0][0].rst_voltage - self.sf_min_input) + self.rst_time + self.fd_reset_time
                if self.pixel_time > 1/input_clk_freq*(2**4):
                    self.pixel_readout_time = self.pixel_time * 2*CNN_kernel*CNN_kernel
                else:
                    self.pixel_readout_time = 1/input_clk_freq*(2**4) * 2*CNN_kernel*CNN_kernel
            else:
                self.pixel_readout_time = 1/input_clk_freq*(2**4) * 2*CNN_kernel*CNN_kernel
            self.readout_circuit_time = input_clk_freq*4 + 1/input_clk_freq*(2**4) + input_clk_freq*8
            self.frame_time = self.exposure_time + self.pixel_readout_time + self.readout_circuit_time

        # Update PGA input frequency to the actual frame rate.
        self.PGA.input_clk_freq = self.frame_rate

        # ==================== SECTION 8: Calculate Power Consumption ====================
        # Calculate the reset power consumption. Since different CFA has different number of colored pixels, the reset power consumption is different.
        # Loop through all the pixels to get their reset power consumption. For different pixel types, the reset power consumption is different, as the different Cap is used for charge collection.
        if Pixel_type == 0 or Pixel_type == 2:
            #3D stacked pixel design
            if photodiode_type == 1 or 2:
                for i in range(3):
                    for j in range(5):
                        self.rst_power = self.compute_rst_energy(self.pixels[i][j].pd_node_cap + self.pixels[i][j].fd_cap)*self.frame_rate*num_cols*num_rows*input_pixel_map[j][0]
            else:
                #For the RGBW #1 CFA design, the number of each color is not 1/4 of the total number of pixels. So the reset power consumption is different.
                if CFA_type != 6:
                    for i in range(4):
                        for j in range(5):
                            self.rst_power = self.compute_rst_energy(self.pixels[i][j].pd_node_cap + self.pixels[i][j].fd_cap)*self.frame_rate*num_cols*num_rows/4*input_pixel_map[j][0]
                else:
                    for j in range(5):
                        self.rst_power = self.compute_rst_energy(self.pixels[0][j].pd_node_cap + self.pixels[0][j].fd_cap)*self.frame_rate*num_cols*num_rows/8
                        self.rst_power += self.compute_rst_energy(self.pixels[1][j].pd_node_cap + self.pixels[1][j].fd_cap)*self.frame_rate*num_cols*num_rows/4
                        self.rst_power += self.compute_rst_energy(self.pixels[2][j].pd_node_cap + self.pixels[2][j].fd_cap)*self.frame_rate*num_cols*num_rows/8
                        self.rst_power += self.compute_rst_energy(self.pixels[3][j].pd_node_cap + self.pixels[3][j].fd_cap)*self.frame_rate*num_cols*num_rows/2
        elif Pixel_type == 1:
            if photodiode_type == 1 or 2:
                for i in range(3):
                    for j in range(5):
                        self.rst_power = self.compute_rst_energy(self.pixels[i][j].Cap_FB + self.pixels[i][j].load_cap)*self.frame_rate*num_cols*num_rows*input_pixel_map[j][0]
            else:
                if CFA_type != 6:
                    for i in range(4):
                        for j in range(5):
                            self.rst_power = self.compute_rst_energy(self.pixels[i][j].Cap_FB + self.pixels[i][j].load_cap)*self.frame_rate*num_cols*num_rows/4*input_pixel_map[j][0]
                else:
                    for j in range(5):
                        self.rst_power = self.compute_rst_energy(self.pixels[0][j].Cap_FB + self.pixels[0][j].load_cap)*self.frame_rate*num_cols*num_rows/8*input_pixel_map[j][0]
                        self.rst_power += self.compute_rst_energy(self.pixels[1][j].Cap_FB + self.pixels[1][j].load_cap)*self.frame_rate*num_cols*num_rows/4*input_pixel_map[j][0]
                        self.rst_power += self.compute_rst_energy(self.pixels[2][j].Cap_FB + self.pixels[2][j].load_cap)*self.frame_rate*num_cols*num_rows/8*input_pixel_map[j][0]
                        self.rst_power += self.compute_rst_energy(self.pixels[3][j].Cap_FB + self.pixels[3][j].load_cap)*self.frame_rate*num_cols*num_rows/2*input_pixel_map[j][0]
        elif Pixel_type == 4:
            if photodiode_type == 1 or 2:
                for i in range(3):
                    for j in range(5):
                        self.rst_power = self.compute_rst_energy(self.pixels[i][j].pd_node_cap)*self.frame_rate*num_cols*num_rows*input_pixel_map[j][0]
            else:
                if CFA_type != 6:
                    for i in range(4):
                        for j in range(5):
                            self.rst_power = self.compute_rst_energy(self.pixels[i][j].pd_node_cap)*self.frame_rate*num_cols*num_rows/4*input_pixel_map[j][0]
                else:
                    for j in range(5):
                        self.rst_power = self.compute_rst_energy(self.pixels[0][j].pd_node_cap)*self.frame_rate*num_cols*num_rows/8
                        self.rst_power += self.compute_rst_energy(self.pixels[1][j].pd_node_cap)*self.frame_rate*num_cols*num_rows/4
                        self.rst_power += self.compute_rst_energy(self.pixels[2][j].pd_node_cap)*self.frame_rate*num_cols*num_rows/8
                        self.rst_power += self.compute_rst_energy(self.pixels[3][j].pd_node_cap)*self.frame_rate*num_cols*num_rows/2
             
        # Source follower power, it include the reset power consumption and the signal readout power consumption.
        self.sf_power = ((self.compute_rst_readout_energy() + self.compute_signal_readout_energy()) *
                        self.frame_rate * num_cols * num_rows / pixel_binning_map[1])
        self.sf_power += self.bias_transistor.bias_current * self.analog_V_dd * num_cols
        
        # ==================== Bus Power Calculation ====================
        # Bus counts, it counts the numbber of buses in different direction and different working rates.
        self.num_of_horizontal_bus_subframe_rate = num_rows * self.in_pixel_latch.num_bit
        self.num_of_horizontal_bus_frame_rate = num_rows * (num_of_tap + 2)
        self.num_of_vertical_bus = num_cols
        if Pixel_type == 0 or Pixel_type == 4:
            self.num_of_vertical_analog_bus = num_of_tap * num_cols
        elif Pixel_type == 1:
            self.num_of_vertical_analog_bus = num_cols * 2 * num_of_tap
        else:
            self.num_of_vertical_analog_bus = 0
        self.num_of_horizontal_analog_bus = 1

        # Bus energy per transition for the bus type mentioned above.
        self.bus_energy_per_transition_horizontal = self.buffer.unit_energy * self.horizontal_bus_length
        self.bus_energy_per_transition_vertical = self.buffer.unit_energy * self.vertical_bus_length
        self.analog_bus_energy_per_transition_horizontal = self.Analog_Buffer.unit_energy * self.horizontal_bus_length
        self.analog_bus_energy_per_transition_vertical = self.Analog_Buffer.unit_energy * self.vertical_bus_length
        
        # Number of samples per frame, for some buses, the sample rate is two, means the it transfer input or output data twice per frame.
        self.num_sample_per_frame = 1
        if CDS_type == 1:
            #For the Coded Exposure design, it need to transfer the data from both type
            if CIS_type == 1:
                self.num_sample_per_frame = num_of_tap

        # For the digital CDS, the sample rate is two, means the it transfer input or output data twice per frame for each tap.
        if CDS_type == 2:
            self.num_sample_per_frame = 2
        
        # Bus power components
        # Subframe rate bus power only for Coded Exposure (CIS_type == 1)
        if CIS_type == 1:
            self.horizontal_bus_power_subframe_rate = (self.num_of_horizontal_bus_subframe_rate *
                                                      self.bus_energy_per_transition_horizontal *
                                                      subframe_rates * frame_rate)
            self.vertical_bus_power_subframe_rate = (self.num_of_vertical_bus *
                                                    self.bus_energy_per_transition_vertical *
                                                    subframe_rates * frame_rate)
        else:
            self.horizontal_bus_power_subframe_rate = 0
            self.vertical_bus_power_subframe_rate = 0
        
        self.horizontal_bus_power_frame_rate = (self.num_of_horizontal_bus_frame_rate *
                                               self.bus_energy_per_transition_horizontal *
                                               self.frame_rate * self.num_sample_per_frame)
        self.vertical_analog_bus_power_frame_rate = (self.num_of_vertical_analog_bus *
                                                     self.analog_bus_energy_per_transition_vertical *
                                                     self.frame_rate * self.num_sample_per_frame)
        
        # Component bus power
        self.PGA_bus_power = (self.bus_energy_per_transition_horizontal * self.frame_rate * num_of_tap *
                             (1 + 2 * PGA_DC_gain))
        self.CDS_bus_power = self.bus_energy_per_transition_horizontal * self.frame_rate * num_of_tap
        self.SH_bus_power = self.bus_energy_per_transition_horizontal * 2 * self.frame_rate * num_of_tap
        self.Mux_bus_power = self.bus_energy_per_transition_horizontal * num_of_tap * num_mux_input * self.frame_rate 

        # Calculate the bus power for different CIS types.
        if CIS_type == 0:
            self.bus_power = self.horizontal_bus_power_frame_rate + self.vertical_analog_bus_power_frame_rate
            self.digital_bus_power = self.horizontal_bus_power_frame_rate
            self.analog_bus_power = self.vertical_analog_bus_power_frame_rate
        elif CIS_type == 1:
            self.bus_power = self.horizontal_bus_power_subframe_rate + self.horizontal_bus_power_frame_rate + self.vertical_bus_power_subframe_rate + self.vertical_analog_bus_power_frame_rate
            self.digital_bus_power = self.horizontal_bus_power_subframe_rate + self.horizontal_bus_power_frame_rate + self.vertical_bus_power_subframe_rate
            self.analog_bus_power = self.vertical_analog_bus_power_frame_rate
        #For the CNN pixel design, the bus power is not take into account.
        elif CIS_type == 2:
            self.bus_power = 0
            self.digital_bus_power = 0
            self.analog_bus_power = 0
        else:
            sys.exit("No such CIS type")
        
        # Add the PGA bus power to the total bus power.
        if PGA_type == 1:
            self.bus_power += self.PGA_bus_power
            self.digital_bus_power += self.PGA_bus_power
        elif PGA_type == 2:
            self.bus_power += self.PGA_bus_power*2
            self.digital_bus_power += self.PGA_bus_power*2
        
        # Add the CDS bus power to the total bus power.
        if CDS_type == 1:
            if CIS_type == 1:
                self.bus_power += self.SH_bus_power
                self.digital_bus_power += self.SH_bus_power
        elif CDS_type == 2:
            self.bus_power += self.CDS_bus_power
            self.digital_bus_power += self.CDS_bus_power
        elif CDS_type != 0:
            sys.exit("No such CDS 2")
        
        # Add the ADC bus power to the total bus power.
        if ADC_type == 1:
            self.bus_power += self.analog_bus_energy_per_transition_horizontal*self.frame_rate
            self.analog_bus_power += self.analog_bus_energy_per_transition_horizontal*self.frame_rate
        elif ADC_type == 2:
            self.bus_power += self.bus_energy_per_transition_horizontal*self.SAR_ADC_input_clk_freq
        elif ADC_type != 0:
            sys.exit("NO such ADC type")

        # Add the MUX bus power to the total bus power.
        if MUX_type == 1:
            self.bus_power += self.Mux_bus_power
            self.digital_bus_power += self.Mux_bus_power
        elif MUX_type != 0:
            sys.exit("No such MUX")
        
        # Calculate the pixel power
        self.pixel_power = (self.rst_power+self.sf_power) * num_of_tap/(pixel_binning_map[0]*pixel_binning_map[1])

        # Calculate the pixel contrl power, it is meanly for the coded exposure design.
        self.in_pixel_ctrl_power = self.in_pixel_ctrl.total_power*num_cols*num_rows
        self.in_pixel_latch_power = self.in_pixel_latch.total_power*num_cols*num_rows

        # Calcualte the Transmission gate power, it is only for the Coded Exposure design. It is actually a part of the control logic power for the CE SRAM.
        if CIS_type == 1:
            self.TG_switch_power = 1/2*self.pixels[0][0].tg.gate_cap*(analog_V_dd**2)*self.subframe_rates_per_second*num_of_tap*num_PD_per_tap
        else:
            self.TG_switch_power = 0
        
        # Calculate the total pixel power for different CIS types.
        if CIS_type == 0:
            self.total_pixel_power = self.pixel_power
        elif CIS_type == 1:
            self.total_pixel_power = self.pixel_power + self.in_pixel_ctrl_power + self.in_pixel_latch_power + self.TG_switch_power
        #For the CNN pixel design, the pixel power is calcualted seperately in the later part of the code.
        elif CIS_type == 2:
            self.total_pixel_power = 0
        else:
            sys.exit("No such CIS type")
        self.per_pixel_power = self.total_pixel_power / (num_cols * num_rows)

        # Pixel type 3 is the PWM pixel based optical CNN pixle design. Refer to the Nature paper of the optical CNN.
        if Pixel_type == 3:
            self.polarity_judge_power = 1/2*self.inv.total_switch_cap*(analog_V_dd**2) * 2 * self.frame_rate * num_rows*num_cols/9
            self.in_pixel_cap_power = 1/2*load_cap*(analog_V_dd**2) * 4 * self.frame_rate * num_cols * num_rows/9
            self.in_pixel_switch_power = 1/2*self.bias_transistor.gate_cap * (analog_V_dd**2) * (14+(2*2)) * self.frame_rate * num_rows * num_cols/9
            self.in_pixel_comparator_power = self.comparator.power*num_cols*num_rows/9
            self.total_pixel_power = self.polarity_judge_power + self.in_pixel_cap_power + self.in_pixel_switch_power + self.in_pixel_comparator_power

        # Calculate the PGA power, it is the power consumption of the PGA circuit.
        self.PGA_power = self.PGA.power
        # Calculate the buffer power, it is the power consumption of the output buffer circuit.
        self.buffer_power = 0.5*self.output_buffer.total_switch_cap*(digital_V_dd**2)*num_cols*num_rows*self.frame_rate
        # Calculate the SH power, it is the power consumption of the SH circuit.
        self.SH_power = (self.SH.energy*self.frame_rate*num_cols + self.buffer_power)
        # Calculate the CDS power, it is the power consumption of the CDS circuit.
        self.CDS_power = self.CDS.total_power
        # Initialize the ADC power to 0, it is calculated seperately in the later part of the code.
        self.ADC_power = 0
        # Initialize the readout total power to 0, it is calculated seperately in the later part of the code.
        self.readout_total_power = 0
        # Calculate the MUX power, it is the power consumption of the MUX circuit. Mainly for the Coded Exposure design.
        self.num_mux = math.ceil(num_cols/num_mux_input)
        self.MUX_power = 1/2*self.mux.capOutput*(analog_V_dd**2)*self.frame_rate*self.num_mux

        # Add the PGA power to the readout total power.
        if PGA_type == 1:
            self.readout_total_power += self.PGA_power
        elif PGA_type == 2:
            self.PGA_power *= 2
            self.readout_total_power += self.PGA_power

        # Add the CDS power to the readout total power.
        if CDS_type == 1:
            if CIS_type == 0:
                self.readout_total_power += self.CDS_power
            elif CIS_type == 1:
                self.readout_total_power += self.SH_power*num_of_tap
        elif CDS_type != 0:
            sys.exit("No such CDS 3")
        
        # Add the ADC power to the readout total power.
        if ADC_type == 1:
            self.ADC_power = (self.SS_ADC.comparator_power/pixel_binning_map[1] + self.SS_ADC.ramp_generator_power/(pixel_binning_map[0]*pixel_binning_map[1]) + self.SS_ADC.counter_total_power*(subframe_rates/max_subframe_rates))*num_of_tap/(pixel_binning_map[0]*pixel_binning_map[1])
            self.readout_total_power += self.ADC_power
        elif ADC_type == 2:
            self.ADC_power = (self.SAR_ADC.comparator_power/pixel_binning_map[1] + self.SAR_ADC.DAC_power*(subframe_rates/max_subframe_rates)/(pixel_binning_map[0]*pixel_binning_map[1]) + self.SAR_ADC.SAR_power*(subframe_rates/max_subframe_rates))*num_of_tap/(pixel_binning_map[0]*pixel_binning_map[1])
            self.readout_total_power += self.ADC_power
        elif ADC_type != 0:
            sys.exit("NO such ADC type")
        
        # Add the MUX power to the readout total power.
        if MUX_type == 1:
            self.readout_total_power += self.MUX_power*num_of_tap
        elif MUX_type != 0:
            sys.exit("No such MUX")

        if Pixel_type == 1 and MUX_type !=1:
            sys.exit("for CTIA pixel, a 2 to 1 mux is required")

        # Add the MIPI power to the readout total power.
        if IO_type == 1:
            self.readout_total_power += self.MIPI.MIPI_power/(pixel_binning_map[0]*pixel_binning_map[1])
        elif IO_type != 0:
            sys.exit("No such I/O")
        # The total readout power is divided by the pixel binning map to get the actual power.
        self.readout_total_power /= pixel_binning_map[1]


        ###Input circuit Power
        ## Input circuit power for Coded Exposure designs
        # Calculate the deserializer power, it is the power consumption of the deserializer circuit.
        self.num_of_deserializer = num_cols/self.deserializer_output_bits
        self.deserializer_power = self.deserializer.total_power * self.num_of_deserializer * subframe_rates/max_subframe_rates

        # Calculate the row logic power, it is the power consumption of the row logic circuit.
        self.num_of_row_logic = num_rows
        self.row_logic_power = self.row_logic.total_power * self.num_of_row_logic * subframe_rates/max_subframe_rates

        # Calculate the time generator power, it is the power consumption of the time generator circuit.
        self.time_generator_power = self.time_generator.total_power
        
        # Calculate the input total power for the Coded Exposure design. For other designs, there is no extra input logics.
        if CIS_type == 1:
            self.input_total_power = self.deserializer_power + self.row_logic_power + self.time_generator_power
        elif if_time_generator == 1:
            self.input_total_power = self.time_generator_power
        else:
            self.input_total_power = 0
        
        # ==================== PLL Power Calculation ====================
        self.pll_power = self.PLL.total_dynamic_power
        self.PFD_power = self.PLL.PFD_dynamic_power
        self.CP_power = self.PLL.CP_dynamic_power
        self.VCO_power = self.PLL.VCO_dynamic_power
        self.FD_power = self.PLL.FD_dynamic_power

        # ==================== Input Driver Power Calculation ====================
        #The input driver power for the inputs in frame rate frequency.
        self.frame_ctrl_input_driver_power = self.frame_ctrl_input_driver.total_power*3
        #The input driver power for the inputs in subframe rate frequency.
        self.Subframe_ctrl_input_driver_power = self.Subframe_ctrl_input_driver.total_power*3

        #The input driver power for the exposure control input. It is used in the Coded Exposure pixel design.
        self.exposure_ctrl_inv_power = 1/2*(self.inv.total_switch_cap)*(analog_V_dd**2)*subframe_rates * num_rows

        # Calculate the input driver total power for the Coded Exposure design. For other designs, there is no extra input logics.
        if CIS_type == 1:
            self.input_driver_total_power = self.frame_ctrl_input_driver_power + self.Subframe_ctrl_input_driver_power+self.exposure_ctrl_inv_power
        else:
            self.frame_ctrl_input_driver_power = self.frame_ctrl_input_driver.total_power*(2+num_PD_per_tap)
            self.input_driver_total_power = self.frame_ctrl_input_driver_power

        # Calculate the system total power by now. It will be updated later.
        self.system_total_power = self.total_pixel_power + self.readout_total_power + self.input_total_power + self.input_driver_total_power + self.bus_power
        
        # Add the PLL power to the system total power.
        if if_PLL == 1:
            self.system_total_power += self.pll_power

        # ==================== Power model for CNN sensor design ====================
        if CIS_type == 2:
            #Pixel type 2 is the TIA pixel based optical CNN pixle design. Refer to the Nature paper of the optical CNN.
            #below is the total power for the CNN pixel design.
            if Pixel_type == 2:
                num_cols = num_cols/CNN_kernel
                num_rows = num_rows/CNN_kernel
                self.pixel_power = self.rst_power + self.PMW_comparator.power * 1/input_clk_freq*(2**4) *num_cols*num_cols*frame_rate/(CNN_kernel**2)
                self.pixel_power += self.TIA.power
            else:
                self.pixel_power = self.rst_power + self.PMW_comparator.power * 1/input_clk_freq*(2**4) *num_cols*num_cols*frame_rate
                
            #Calcualte the total power for the CNN analog logic
            self.ramp_generator_power = (self.PMW_ramp_generator.charge_energy+self.PMW_ramp_generator.discharge_energy)*frame_rate*2*CNN_kernel*CNN_kernel*num_rows/(2*CNN_kernel) + (self.readout_ramp_generator.charge_energy+self.readout_ramp_generator.discharge_energy)*frame_rate*num_rows/(2*CNN_kernel)
            self.polarity_judge_power = 1/2*self.inv.total_switch_cap*(analog_V_dd**2) * 2 * frame_rate * num_rows*num_cols/(CNN_kernel**2)
            self.CNN_logic_cap_power = 1/2*load_cap*(analog_V_dd**2) * 4 * frame_rate * num_cols * num_rows/(CNN_kernel**2)
            self.CNN_logic_switch_power = 1/2*self.bias_transistor.gate_cap * (analog_V_dd**2) * (14+(2*2)) * frame_rate * num_rows * num_cols/(CNN_kernel**2)
            self.CNN_logic_comparator_power = self.comparator.power*num_cols/CNN_kernel
            self.CNN_logic_power = self.polarity_judge_power + self.CNN_logic_cap_power + self.CNN_logic_switch_power + self.CNN_logic_comparator_power
            
            #Calcualte the total power for the CNN digital logic
            self.or_gate_power = (self.nor.total_switch_cap + self.inv.total_switch_cap)*(analog_V_dd**2)*frame_rate*num_cols*num_rows/(4*(CNN_kernel**2))
            self.and_gate_power = (self.nand.total_switch_cap + self.inv.total_switch_cap)*(analog_V_dd**2)*frame_rate*4*num_rows*num_cols/(4*(CNN_kernel**2))
            self.counter_power = self.CNN_case_counter.total_power
            self.B2T_power = 1/2*(self.nand.total_switch_cap + 5*self.inv.total_switch_cap + 3*self.nor.total_switch_cap)*(analog_V_dd**2)*frame_rate*4*num_cols*num_rows/(4*(CNN_kernel**2))
            self.up_down_counter_power = self.up_down_counter.total_power
            self.readout_total_power = self.or_gate_power + self.and_gate_power + self.counter_power + self.B2T_power + self.up_down_counter_power

            #Calcualte the total power for the CNN bus power
            self.CNN_logic_bus_power = self.bus_energy_per_transition_horizontal * 10 * num_rows/3
            self.Readout_bus_power = self.bus_energy_per_transition_horizontal * 3
            self.Analog_ramp_bus_power = self.analog_bus_energy_per_transition_horizontal * 2 #two ramp generator
            self.bus_power = self.CNN_logic_bus_power + self.Readout_bus_power

            #Calculate the total system power for the CNN sensor design.
            self.system_total_power = (self.pixel_power + self.ramp_generator_power + self.CNN_logic_power +
                                     self.readout_total_power + self.bus_power)

        # ==================== Calculate Noise ====================
        # Update pixel bias current
        for i in range(4):
            for j in range(5):
                self.pixels[i][j].bias_current = self.bias_current

        # Calculate ADC input capacitance, used for the noise model to calculate the noise.
        if ADC_type == 1:
            self.ADC_input_cap = (self.load_cap + self.pixels[0][0].sf.drain_cap +
                                self.bias_transistor.drain_cap + self.SS_ADC.comparator.input_cap)
        elif ADC_type == 2:
            self.ADC_input_cap = (self.load_cap + self.pixels[0][0].sf.drain_cap +
                                self.bias_transistor.drain_cap + self.SAR_ADC.comparator.input_cap)
        else:
            self.ADC_input_cap = 1

        # PD load cap is an input parameter to the ADC noise model, it is used to calcualte the conversion gain.
        if Pixel_type == 4:
            self.PD_load_cap = self.pixels[0][0].pd_capacitance
        elif Pixel_type == 1:
            self.PD_load_cap = self.pixels[0][0].Cap_FB
        else:
            self.PD_load_cap = self.pixels[0][0].fd_cap

        #Intialize the ADC noise model, it is used to calculate the quantization noise.
        self.ADC_Noise_model = ADC_Quantizing_Noise(
            ADC_resolution=adc_resolution,
            voltage_range=self.digital_V_dd-self.min_output_voltage,
            cap=self.PD_load_cap
        )

        #Intialize the ADC thermal noise model, it is used to calculate the thermal noise.
        self.ADC_thermal_noise_model = Thermal_Noise(
            transconductance=self.SS_ADC.comparator.Gm,
            cap=self.PD_load_cap,
            temperature=300
        )

        # Calculate reset capacitance of different pixel types for noise model. It includes all the capacitance need to be reset
        if Pixel_type == 0:
            self.rst_cap = (self.pixels[0][0].fd_cap + self.pixels[0][0].pd_capacitance +
                          self.pixels[0][0].tg.source_cap)
        elif Pixel_type == 1:
            self.rst_cap = self.pixels[0][0].Cap_FB + self.pixels[0][0].load_cap
        elif Pixel_type == 2:
            self.rst_cap = (CNN_kernel**2 * self.pixels[0][0].fd_cap + self.pixels[0][0].pd_capacitance +
                          self.pixels[0][0].tg.source_cap)
        else:
            self.rst_cap = self.pixels[0][0].pd_capacitance + self.pixels[0][0].tg.source_cap
        
        #Intialize the reset noise model, it is used to calculate the reset noise.
        self.Reset_Noise_model = Reset_Noise(cap=self.rst_cap, temperature=300)

        # Extract noise values from the noise models.
        self.reset_noise = self.Reset_Noise_model.reset_noise
        self.ADC_quantization_noise_square = self.ADC_Noise_model.quant_noise_square
        self.ADC_thermal_noise_square = self.ADC_thermal_noise_model.thermal_noise_square
        
        # Update pixel parameters
        for i in range(4):
            for j in range(5):
                self.pixels[i][j].sf_vds = (self.analog_V_dd - 0.5*(self.max_output_voltage + self.min_output_voltage))
                self.pixels[i][j].load_cap = self.load_cap

        # Extract noise values from the pixel noise models, which are defined in the pixel class.
        self.photo_shot_noise_square = self.pixels[0][0].shot_noise_model.shot_noise_square
        self.sf_shot_noise_square = self.pixels[0][0].current_shot_noise_model.current_shot_noise_square
        self.sf_thermal_noise_square = self.pixels[0][0].sf_thermal_noise_model.thermal_noise_square
        self.tg_thermal_noise_square = self.pixels[0][0].tg_thermal_noise_model.thermal_noise_square

        # Calculate the total thermal noise, it is the sum of the thermal noise from the source follower, transmission gate and ADC.
        if ADC_type != 0:
            self.total_thermal_noise_square = (self.sf_thermal_noise_square + self.tg_thermal_noise_square +
                                              self.ADC_thermal_noise_square)
        else:
            self.total_thermal_noise_square = self.sf_thermal_noise_square + self.tg_thermal_noise_square
        
        # Extract noise values from the pixel noise models, which are defined in the pixel class.
        self.dark_current_noise_square = self.pixels[0][0].dark_current_noise_model.dark_current_noise_square
        self.transfer_noise_square = self.pixels[0][0].transfer_noise_model.transfer_noise_square
        
        # Calculate the total shot noise, it is the sum of the shot noise from the photodiode, source follower and dark current.
        self.total_shot_noise_square = ((self.photo_shot_noise_square + self.sf_shot_noise_square +
                                       self.dark_current_noise_square) *
                                      pixel_binning_map[0] * pixel_binning_map[1])
        
        # Calculate the total read noise, it is the sum of the thermal noise and the quantization noise.
        if ADC_type != 0:
            self.total_read_noise_square = self.total_thermal_noise_square + self.ADC_quantization_noise_square
        else:
            self.total_read_noise_square = self.total_thermal_noise_square

        # Calculate the total noise, it is the sum of the shot noise, read noise and transfer noise.
        self.total_noise_square = (self.total_shot_noise_square + self.total_read_noise_square +
                                  self.transfer_noise_square)
        
        # Calculate SNR, DR, and FWC for each color and pixel size
        self.SNR = []
        self.DR = []
        self.FWC = []
        for i in range(4):
            local_SNR = []
            local_DR = []
            local_FWC = []
            for j in range(5):
                # Calculate the signal level, it is the "S" in the SNR equation.
                signal = (self.pixels[i][j].shot_noise_model.shot_noise_square *
                         pixel_binning_map[0] * pixel_binning_map[1] * subframe_rates/max_subframe_rates)
                # Calculate the noise level, it is the "N" in the SNR equation.
                SNR_noise_square = (self.pixels[i][j].shot_noise_model.shot_noise_square *
                                  subframe_rates/max_subframe_rates + self.total_read_noise_square +
                                  self.transfer_noise_square)
                # Calculate the SNR for each pixel
                local_SNR.append(20 * math.log10(signal / math.sqrt(SNR_noise_square)))
                # Calculate the FWC for each pixel
                FWC_temp = (self.pixels[i][j].PD.FWC * num_PD_per_tap *
                          pixel_binning_map[0] * pixel_binning_map[1])
                local_FWC.append(FWC_temp)
                # Calculate the DR for each pixel
                local_DR.append(20 * math.log10(FWC_temp / math.sqrt(self.total_read_noise_square)))

            # Append the SNR, DR and FWC for each pixel to the global lists.
            self.SNR.append(local_SNR)
            self.DR.append(local_DR)
            self.FWC.append(local_FWC)
        

        # Print results
        if self.print_output:
            self.print_timing_results(frame_rate, Pixel_type, CIS_type, PGA_type, CDS_type, ADC_type, IO_type)
            self.print_power_consumption(CIS_type, Pixel_type, PGA_type, CDS_type, ADC_type, MUX_type, IO_type, if_PLL, pixel_binning_map)
            self.print_noise_results(ADC_type)

    # ==================== Calculation Helper Methods ====================
    
    def compute_rst_energy(self, total_cap):
        """Calculate reset energy for given capacitance"""
        energy = total_cap*self.pixels[0][0].rst_voltage*self.analog_V_dd
        return energy

    def compute_signal_readout_energy(self):
        """Calculate signal readout energy"""
        energy = self.bias_current * self.analog_V_dd * self.readout_time
        return energy

    def compute_rst_readout_energy(self):
        """Calculate reset readout energy"""
        energy = self.bias_current * self.analog_V_dd * self.readout_time
        return energy
    
    def compute_bias_current(self):
        """Calculate bias current for source follower"""
        bias_current = (0.5 * self.bias_transistor.un * self.bias_transistor.Cox *
                       (self.bias_transistor.width / self.bias_transistor.length) *
                       ((self.bias_voltage - self.bias_transistor.v_th) ** 2) *
                       (1 + self.bias_transistor.lambda_param * self.bias_voltage))
        return bias_current
    
    def compute_readout_time(self, max_voltage, min_voltage):
        """
        Compute the readout time
        
        Args:
            max_voltage: max possible voltage of the output cap
            min_voltage: min possible voltage of the output cap
        """
        # Source follower output resistance (1/gm)
        sf_output_resistance = (1 / math.sqrt(2 * self.pixels[0][0].sf.un * self.pixels[0][0].sf.Cox *
                                            (self.pixels[0][0].sf.width / self.pixels[0][0].sf.length) *
                                            self.bias_current))
        bias_transistor_resistance = 1 / (self.bias_transistor.lambda_param * self.bias_current)
        output_resistance = ((sf_output_resistance * bias_transistor_resistance) /
                           (sf_output_resistance + bias_transistor_resistance))

        # Total capacitance
        total_capacitance = (self.pixels[0][0].sf.source_cap +
                           self.output_bus.cap_wire_per_m * self.output_bus_length +
                           self.load_cap + self.PGA.input_cap +
                           self.PGA.comparator.input_transistor.gate_cap)
        total_resistance = output_resistance + self.output_bus.res_wire_per_m * self.output_bus_length

        # Calculate slew and settle times
        t_slew = (max_voltage - min_voltage) * total_capacitance / self.bias_current
        t_settle = total_resistance * total_capacitance * 5  # K_settle = 5 for stable settling
        readout_time = t_slew + t_settle
        return readout_time
    
    def compute_sf_input_voltage(self, sf_output_voltage):
        """Calculate source follower input voltage from output voltage"""
        Vgs = (math.sqrt((2 * self.bias_current) /
                        (self.pixels[0][0].sf.un * self.pixels[0][0].sf.Cox *
                         self.pixels[0][0].sf.width / self.pixels[0][0].sf.length)) +
              self.pixels[0][0].sf.v_th)
        sf_input_voltage = sf_output_voltage + Vgs
        return sf_input_voltage
    
    def compute_sf_out_voltage(self, sf_input_voltage):
        """Calculate source follower output voltage from input voltage"""
        Vgs = (math.sqrt((2 * self.bias_current) /
                        (self.pixels[0][0].sf.un * self.pixels[0][0].sf.Cox *
                         self.pixels[0][0].sf.width / self.pixels[0][0].sf.length)) +
              self.pixels[0][0].sf.v_th)
        sf_output_voltage = sf_input_voltage - Vgs
        return sf_output_voltage

    def compute_rc_lpf(self, reference_freq_hz, divider=10):
        """Calculate RC values for low-pass filter"""
        fc = reference_freq_hz / divider
        rc = 1 / (2 * math.pi * fc)
        R = 1e3  # Ohms
        C = rc / R  # Farads
        return R, C
    
    # ==================== Output/Printing Methods ====================

    def print_timing_results(self, frame_rate, Pixel_type, CIS_type, PGA_type, CDS_type, ADC_type, IO_type):
        """Print timing model results"""
        if CIS_type == 2:
            print(f"========= TIMING MODEL RESULTS =========")
            print(f"Frame Time : {self.format_time(self.frame_time)}")
            print(f"  └─ Exposure Time : {self.format_time(self.exposure_time)}")
            print(f"  └─ Pixel Readout Time : {self.format_time(self.pixel_readout_time)}")
            print(f"  └─ CNN Processing Time : {self.format_time(self.readout_circuit_time)}")
            print(f"Frame Rate : {frame_rate} Hz")
            print(f"  └─ Max Frame Rate : {1/self.frame_time} Hz")
        else:
            print(f"Num of Effective pixels : {self.effective_pixels}")
            print(f"========= TIMING MODEL RESULTS =========")
            print(f"Frame Time : {self.format_time(self.frame_time)}")
            print(f"  └─ Exposure Time : {self.format_time(self.exposure_time)}")
            print(f"  └─ Readout Time : {self.format_time(self.readout_time)}")
            if PGA_type == 1:
                print(f"  └─ PGA Time : {self.format_time(self.PGA.latency)}")
            elif PGA_type == 2:
                print(f"  └─ PGA Time : {self.format_time(self.PGA.latency*2)}")
            if CDS_type == 1:
                if CIS_type == 1:
                    print(f"  └─ CDS Time : {self.format_time(self.SH.delay)}")
            if CDS_type == 2:
                print(f"  └─ CDS Time : {self.format_time(self.CDS.CDS_delay)}")
            if IO_type == 1:
                print(f"  └─ I/O Time : {self.format_time(self.MIPI_delay)}")
            print(f"  └─ ADC Time : {self.format_time(self.sel_time)}")
            print(f"  └─ Idle Time : {self.format_time(self.idle_time)}")
            print(f"  └─ Pixel Reset Time : {self.format_time(self.rst_time)}")
            print(f"  └─ Pixel Readout Time : {self.format_time(self.readout_time)}")
            if CIS_type == 1:
                print(f"  └─ Subframe Time : {self.format_time(self.subframe_time)}")
            if Pixel_type == 0:
                print(f"  └─ Pixel Floating Diffusion Reset Time : {self.format_time(self.fd_reset_time)}")
            print(f"Max Exposure Time : {self.format_time(self.max_exposure_time)}")
            if CIS_type == 1:
                print(f"Max Subframes (Calc): {self.max_subframe_rates:10.0f}")
                print(f"Actual Subframes : {self.subframe_rates:10.0f}")
            print(f"Frame Rate : {self.frame_rate} HZ")
            print(f"Max Frame Rate : {self.max_frame_rate:10.2f} Hz")
            print("-" * 40)

    def print_power_consumption(self, CIS_type, Pixel_type, PGA_type, CDS_type, ADC_type, MUX_type, IO_type, if_PLL, pixel_binning_map):
        """Print power consumption results"""
        print(f"========= POWER CONSUMPTION (W) ========")
        print(f"System Total Power : {self.format_power(self.system_total_power)}")
        print("-" * 40)
        if CIS_type == 2:
            print(f"Pixel Power : {self.format_power(self.pixel_power)}")
            if Pixel_type == 2:
                print(f"└─In-Pixel TIA Power : {self.format_power(self.TIA.power)}")
            print(f"Ramp Generator Power : {self.format_power(self.ramp_generator_power)}")
            print(f"CNN logic Power : {self.format_power(self.CNN_logic_power)}")
            print(f"  └─ Polarity Judge Power : {self.format_power(self.polarity_judge_power)}")
            print(f"  └─ Cap Power : {self.format_power(self.CNN_logic_cap_power)}")
            print(f"  └─ Comparator Power : {self.format_power(self.CNN_logic_comparator_power)}")
            print(f"  └─ Switch Power : {self.format_power(self.CNN_logic_switch_power)}")
            print(f"Readout Circuit Power : {self.format_power(self.readout_total_power)}")
            print(f"  └─ Or Gate Power : {self.format_power(self.or_gate_power)}")
            print(f"  └─ And Gate Power : {self.format_power(self.and_gate_power)}")
            print(f"  └─ Readout Counter Power : {self.format_power(self.counter_power)}")
            print(f"  └─ B2T Power : {self.format_power(self.B2T_power)}")
            print(f"  └─ Up Down Counter Power : {self.format_power(self.up_down_counter_power)}")
            print(f"Bus Power (Total) : {self.format_power(self.bus_power)}")
        else:
            print(f"Pixel Array Power : {self.format_power(self.total_pixel_power)}")
            print(f"  └─ Per-Pixel Power : {self.format_power(self.per_pixel_power)}")
            print(f"  └─ Pixel Readout Power : {self.format_power(self.pixel_power)}")
            print(f"  └─ Pixel Reset Power : {self.format_power(self.rst_power)}")
            print(f"  └─ Pixel Source Follower Power : {self.format_power(self.sf_power)}")
            if CIS_type == 1:
                print(f"  └─ In-Pixel Ctrl Power : {self.format_power(self.in_pixel_ctrl_power)}")
                print(f"  └─ In-Pixel Latch Power: {self.format_power(self.in_pixel_latch_power)}")
                print(f"  └─ TG Switch Power: {self.format_power(self.TG_switch_power)}")
            print("-" * 40)
            print(f"Readout Circuit Power : {self.format_power(self.readout_total_power)}")
            if PGA_type != 0:
                print(f"  └─ PGA Power : {self.format_power(self.PGA_power)}")
            if CDS_type == 1:
                if (CIS_type == 0 and ADC_type != 1) or Pixel_type == 4:
                    print(f"  └─ CDS Power : {self.format_power(self.CDS_power)}")
                elif CIS_type == 1:
                    print(f"  └─ SH/CDS Power : {self.format_power(self.SH_power)}")
            if ADC_type == 1:
                print(f"  └─ SS ADC Power : {self.format_power(self.ADC_power)}")
            elif ADC_type == 2:
                print(f"  └─ SAR ADC Power : {self.format_power(self.ADC_power)}")
            if MUX_type == 1:
                print(f"  └─ MUX Power : {self.format_power(self.MUX_power)}")
            if IO_type == 1:
                print(f"  └─ MIPI Power : {self.format_power(self.MIPI.MIPI_power/(pixel_binning_map[0]*pixel_binning_map[1]))}")
            print("-" * 40)
            if CIS_type == 1:
                print(f"Input Circuit Power : {self.format_power(self.input_total_power)}")
                print(f"  └─ Deserializer Power : {self.format_power(self.deserializer_power)}")
                print(f"  └─ Row Logic Power : {self.format_power(self.row_logic_power)}")
                print(f"  └─ Timing Gen Power : {self.format_power(self.time_generator_power)}")
                print("-" * 40)
            if if_PLL == 1:
                print(f"Phase Locked Loop Power : {self.format_power(self.PLL.total_dynamic_power)}")
                print(f"  └─ PFD Power : {self.format_power(self.PLL.PFD_dynamic_power)}")
                print(f"  └─ CP Power : {self.format_power(self.PLL.CP_dynamic_power)}")
                print(f"  └─ VCO Power : {self.format_power(self.PLL.VCO_dynamic_power)}")
                print(f"  └─ FD Power : {self.format_power(self.PLL.FD_dynamic_power)}")
            print(f"Input Driver Power : {self.format_power(self.input_driver_total_power)}")
            print(f"  └─ Frame Ctrl Driver : {self.format_power(self.frame_ctrl_input_driver_power)}")
            if CIS_type == 1:
                print(f"  └─ Subframe Ctrl Driver: {self.format_power(self.Subframe_ctrl_input_driver_power)}")
                print(f"  └─ Exposure Ctrl INV : {self.format_power(self.exposure_ctrl_inv_power)}")
            print("-" * 40)
            print(f"Bus Power (Total) : {self.format_power(self.bus_power)}")
            print(f"  └─ Bus Power (Digital): {self.format_power(self.digital_bus_power)}")
            print(f"  └─ Bus Power (Analog): {self.format_power(self.analog_bus_power)}")
            print(f"--- Detailed Bus Power Components ---")
            if CIS_type == 1:
                print(f"H-Bus (Subframe Rate) : {self.format_power(self.horizontal_bus_power_subframe_rate)}")
            print(f"H-Bus (Frame Rate) : {self.format_power(self.horizontal_bus_power_frame_rate)}")
            if CIS_type == 1:
                print(f"V-Bus (Subframe Rate) : {self.format_power(self.vertical_bus_power_subframe_rate)}")
            print(f"V-Bus (Analog) : {self.format_power(self.vertical_analog_bus_power_frame_rate)}")
            print("========================================")

    def print_noise_results(self, ADC_type):
        """Print noise analysis results"""
        print("========= Noise =========")
        print(f"Dynamic range       ")
        for i in range(4):
            for j in range(5):
                print(f"    {i}th Color, {j}th pixel size  : {self.DR[i][j]:.2f} dB")
        print(f"Signal Noise Ratio ")
        for i in range(4):
            for j in range(5):
                print(f"    {i}th Color, {j}th pixel size  : {self.SNR[i][j]:.2f} dB")
        for i in range(4):
            for j in range(5):
                print(f"    {i}th Color, {j}th pixel size  : {self.FWC[i][j]} e-")
        print(f"\nTotoal Noise      : {math.sqrt(self.total_noise_square):.2f} e-")
        print(f"  1. Total Shot Noise     : {math.sqrt(self.total_shot_noise_square):.2f} e-")
        print(f"    ├─ Photo Shot Noise       : {math.sqrt(self.photo_shot_noise_square):.2f} e-")
        print(f"    ├─ SF Shot Noise          : {math.sqrt(self.sf_shot_noise_square):.2f} e-")
        print(f"    └─ Dark Current Shot Noise: {math.sqrt(self.dark_current_noise_square):.2f} e-")
        print(f"  2. Total Read Noise     : {math.sqrt(self.total_read_noise_square):.2f} e-")
        print(f"    ─ Total Thermal Noise     : {math.sqrt(self.total_thermal_noise_square *1e6):.2f} µe-")
        print(f"       ├─ SF Thermal Noise    : {math.sqrt(self.sf_thermal_noise_square*1e6):.2f} µe-")
        print(f"       ├─ TG Thermal Noise    : {math.sqrt(self.tg_thermal_noise_square*1e6):.2f} µe-")
        if ADC_type != 0:
            print(f"       └─ ADC Thermal Noise   : {math.sqrt(self.ADC_thermal_noise_square*1e6):.2f} µe-")
            print(f"    ─ ADC Quantization Noise  : {math.sqrt(self.ADC_quantization_noise_square*1e6):.2f} µe-")
        print(f"  3. Transfer(CTI) Noise  : {math.sqrt(self.transfer_noise_square):.2f} e-")


    def format_time(self, value):
        """Format time value for display (ms or µs)"""
        if value >= 1e-3:
            return f"{value * 1e3:10.2f} ms"
        else:
            return f"{value * 1e6:10.2f} µs"
    
    def format_power(self, value):
        """Format power value for display (mW or µW)"""
        if value >= 1e-3:
            return f"{value * 1e3:10.2f} mW"
        else:
            return f"{value * 1e6:10.2f} µW"