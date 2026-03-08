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
import sys
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
from Internal_buffer import Repeater
from Input_Driver import Input_Driver
from analog_buffer import Analog_Buffer, Analog_Buffer_Bus
from MIPI import MIPI
from SRAM_Buffer import SRAM, MUX


# CNN CIS Array class
# Models a CNN-based CMOS image sensor array with integrated processing
# 
# Inputs: Similar structure to CIS_Array with CNN-specific parameters - photodiode_type, num_rows, num_cols, pd_E,
#         Pixel_type, MUX_type, IO_type, input_clk_freq, PLL_output_frequency, CNN_kernel (CNN kernel configuration),
#         and other parameters (see CIS_Array for details)
# Outputs: frame_time (frame time in s), max_frame_rate (maximum frame rate in fps), total_power (total system power in W),
#          pixel_power (pixel array power in W), adc_power (ADC power in W), CNN processing metrics (CNN-related power and timing),
#          total_noise_square (total noise variance in electrons²), SNR (signal-to-noise ratio in dB),
#          DR (dynamic range in dB), and various timing and power metrics
class CNN_CIS_Array:
    """
    CIS Array class specifically for CIS_type = 2 (CNN processing path)
    This class handles CNN-specific timing, power, and noise calculations
    """
    def __init__(self, photodiode_type, num_rows, num_cols, pd_E, Pixel_type, MUX_type, IO_type, input_clk_freq, 
                 PLL_output_frequency, num_PD_per_tap, PGA_type, CDS_type, CFA_type, input_pixel_map, 
                 pixel_binning_map, ADC_type, CDS_amp_gain, additional_latency, num_of_unused_tap, 
                 analog_V_dd, digital_V_dd, adc_resolution, PGA_DC_gain, feature_size_nm, frame_rate, 
                 subframe_rates, max_subframe_rates, CNN_kernel, exposure_time, light_source=1,
                 bias_voltage=0.6, load_cap=100e-15, ramp_bias_voltage=0.6, ramp_load_cap=100e-15,
                 comparator_bias_voltage=0.6, comparator_load_cap=100e-15, PGA_input_cap=100e-15, 
                 comparator_input_cap=100e-15, charge_pump_load_cap=100e-15, V_ctrl=0.6, 
                 ADC_input_clk_freq=0, MIPI_unit_cap=0, if_time_generator=0, PD_saturation_level=0.6, 
                 PD_tech=0, num_of_tap=2, if_PLL=1, num_mux_input=48, deserializer_output_bits=16, 
                 CTIA_C_FB=100e-15, CTIA_load_cap=100e-15, capm2=None, EQE=None, lambda_m=None, 
                 print_output=True):
        
        self.print_output = print_output
        self.ADC_input_freq = ADC_input_clk_freq if ADC_input_clk_freq != 0 else input_clk_freq
        self.subframe_rates = subframe_rates
        self.analog_V_dd = analog_V_dd
        self.digital_V_dd = digital_V_dd
        self.load_cap = load_cap
        self.output_bus_length = num_rows * 1e-6  # in meter
        self.bias_voltage = bias_voltage
        self.deserializer_output_bits = deserializer_output_bits
        self.photodiode_type = photodiode_type
        self.pixel_binning_map = pixel_binning_map
        self.CNN_kernel = CNN_kernel

        if num_PD_per_tap != 1 and Pixel_type != 0:
            sys.exit("only normal pixel type support multiple PD binning")

        # CFA color mapping
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
        
        if photodiode_type != 0:
            self.effective_pixels = num_cols*num_rows
            num_cols = num_cols * 3
            self.color = [0,1,2,0]
        elif photodiode_type == 0:
            if CFA_type in CFA_COLORS:
                self.color = CFA_COLORS[CFA_type]
                self.effective_pixels = num_cols*num_rows/4 if CFA_type != 7 else num_cols*num_rows
            else:
                sys.exit("Error: no such color filter")
        else:
            sys.exit("Error: no such photodiode")

        # Basic components
        self.inv = INV(feature_size_nm=feature_size_nm, V_dd=analog_V_dd)
        self.bias_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )
        self.bias_current = self.compute_bias_current()

        # BUS Modeling
        if num_rows > 100:
            self.bus_wire_type = "global_aggressive"
        else:
            self.bus_wire_type = "semi_aggressive"

        self.horizontal_bus_length = num_cols * 11.2e-6
        self.vertical_bus_length = num_rows * 11.2e-6

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

        self.Analog_Buffer = Analog_Buffer_Bus(
            wire_unit_cap=self.input_bus.cap_wire_per_m,
            wire_unit_res=self.input_bus.res_wire_per_m,
            feature_size_nm=feature_size_nm,
            V_dd=self.analog_V_dd,
            bias_voltage=self.bias_voltage
        )

        self.buffer = Repeater(
            feature_size_nm=feature_size_nm,
            capWirePerUnit=self.input_bus.cap_wire_per_m,
            resWirePerUnit=self.input_bus.res_wire_per_m,
            V_dd=self.analog_V_dd
        )

        self.PGA = PGA(
            bias_voltage=bias_voltage,
            input_clk_freq=frame_rate,
            input_cap=PGA_input_cap,
            DC_gain=PGA_DC_gain,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            frame_rate=frame_rate,
            num_cols=num_cols,
            num_rows=num_rows
        )

        self.TIA = TIA(
            bias_voltage=bias_voltage,
            input_clk_freq=frame_rate,
            input_cap=100e-15,
            DC_gain=2000,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            num_cols=num_cols/CNN_kernel
        )

        # Pixel initialization
        V_swing = PD_saturation_level
        self.pixels = []
        
        common_params = {
            'num_PD_per_tap': num_PD_per_tap,
            'photodiode_type': self.photodiode_type,
            'pd_E': pd_E,
            'feature_size_nm': feature_size_nm,
            'light_source': light_source,
            'bias_current': self.bias_current,
            'V_dd': analog_V_dd,
            'V_swing': V_swing,
            'exposure_time': exposure_time,
            'PD_tech': PD_tech,
            'capm2': capm2,
            'EQE': EQE,
            'lambda_m': lambda_m
        }
        
        for i in range(4):
            temp_pixel_set = []
            for j in range(5):
                if Pixel_type == 0 or Pixel_type == 4:
                    temp_pixel_instance = APS(
                        color=self.color[i],
                        pd_length=input_pixel_map[j][1],
                        pd_width=input_pixel_map[j][2],
                        num_of_tap=num_of_tap,
                        Pixel_type=Pixel_type,
                        **common_params
                    )
                elif Pixel_type == 1:
                    temp_pixel_instance = CTIA(
                        color=self.color[i],
                        pd_length=input_pixel_map[j][1],
                        pd_width=input_pixel_map[j][2],
                        num_of_tap=num_of_tap + num_of_unused_tap,
                        Cap_FB=CTIA_C_FB,
                        load_cap=CTIA_load_cap,
                        **common_params
                    )
                elif Pixel_type == 2:
                    temp_pixel_instance = CNN_PD(
                        color=self.color[i],
                        pd_length=input_pixel_map[j][1],
                        pd_width=input_pixel_map[j][2],
                        **common_params
                    )
                else:
                    sys.exit("NO Such Pixel type")
                temp_pixel_set.append(temp_pixel_instance)
            self.pixels.append(temp_pixel_set)

        # CNN-specific components
        self.mux = MUX(
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            num_input=num_mux_input,
            next_stage_input_Cap=0
        )

        self.comparator = Comparator(
            bias_voltage=comparator_bias_voltage,
            load_cap=comparator_load_cap,
            input_cap=comparator_input_cap,
            DC_gain=2000,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd
        )

        self.PMW_comparator = Comparator(
            bias_voltage=comparator_bias_voltage,
            load_cap=comparator_load_cap,
            input_cap=comparator_input_cap,
            DC_gain=2000,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd
        )

        self.PMW_ramp_generator = Ramp_generator(
            bias_voltage=ramp_bias_voltage,
            load_cap=ramp_load_cap,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            input_clk_freq=input_clk_freq,
            adc_resolution=4
        )

        self.readout_ramp_generator = Ramp_generator(
            bias_voltage=ramp_bias_voltage,
            load_cap=ramp_load_cap,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            input_clk_freq=input_clk_freq,
            adc_resolution=4
        )

        # ADC components (needed for PLL calculation)
        self.min_output_voltage = self.bias_voltage - self.bias_transistor.v_th
        self.max_output_voltage = self.compute_sf_out_voltage(self.pixels[0][0].rst_voltage)
        
        self.CDS = CDS(
            V_rst=self.max_output_voltage,
            V_sig=self.min_output_voltage,
            comaprator_FB_cap=CDS_amp_gain*comparator_input_cap,
            comparator_bias_voltage=comparator_bias_voltage,
            comparator_load_cap=comparator_load_cap,
            comparator_input_cap=comparator_input_cap,
            num_cols=num_cols,
            num_rows=num_rows,
            frame_rate=frame_rate,
            feature_size_nm=feature_size_nm,
            V_dd=analog_V_dd,
            DC_gain=2000
        )
        
        self.SAR_ADC_input_clk_freq = self.ADC_input_freq
        self.SAR_ADC = SAR_ADC(
            max_output_voltage=self.CDS.CDS_output_voltage,
            adc_resolution=adc_resolution,
            comparator_bias_voltage=comparator_bias_voltage,
            comparator_load_cap=comparator_load_cap,
            comparator_input_cap=comparator_input_cap,
            num_cols=num_cols,
            num_rows=num_rows,
            frame_rate=frame_rate,
            input_clk_freq=self.SAR_ADC_input_clk_freq,
            feature_size_nm=feature_size_nm,
            V_dd=digital_V_dd,
            DC_gain=2000
        )

        self.SS_ADC = SS_ADC(
            ramp_bias_voltage=ramp_bias_voltage,
            ramp_load_cap=ramp_load_cap,
            input_clk_freq=self.ADC_input_freq,
            max_output_voltage=self.digital_V_dd,
            adc_resolution=adc_resolution,
            comparator_bias_voltage=comparator_bias_voltage,
            comparator_load_cap=comparator_load_cap,
            comparator_input_cap=comparator_input_cap,
            num_cols=num_cols,
            num_rows=num_rows,
            feature_size_nm=feature_size_nm,
            V_dd=digital_V_dd,
            frame_rate=frame_rate
        )

        # LPF for PLL
        self.LPF_resistance, self.LPF_load_cap = self.compute_rc_lpf(input_clk_freq, divider=10)
        if ADC_type == 1:
            self.PLL_output_frequency = self.SS_ADC.counter_clk_freq
        else:
            self.PLL_output_frequency = PLL_output_frequency

        self.PLL = Phase_lock_loop(
            CP_load_cap=charge_pump_load_cap,
            LPF_resistance=self.LPF_resistance,
            LPF_load_cap=self.LPF_load_cap,
            input_clk_freq=input_clk_freq,
            output_clk_frequency=self.PLL_output_frequency,
            bias_votlage=self.bias_voltage,
            feature_size_nm=feature_size_nm,
            V_dd=self.digital_V_dd,
            V_ctrl=V_ctrl
        )

        self.SH = SHCircuit(
            V_dd=analog_V_dd,
            load_cap=load_cap,
            feature_size_nm=feature_size_nm
        )

        self.nor = NOR(feature_size_nm=feature_size_nm, V_dd=analog_V_dd)
        self.nand = NAND(feature_size_nm=feature_size_nm, V_dd=analog_V_dd)

        self.CNN_case_counter = Countner(
            ADC_resolution=4,
            input_clk_freq=num_cols*num_rows/(4*(CNN_kernel**2)) * frame_rate*4,
            feature_size_nm=feature_size_nm,
            V_dd=digital_V_dd
        )

        self.up_down_counter = Countner(
            ADC_resolution=15,
            input_clk_freq=num_cols*num_rows/(4*(CNN_kernel**2)) * 4,
            feature_size_nm=feature_size_nm,
            V_dd=digital_V_dd
        )

        # Timing calculations

        pixel_rst_time = []
        for i in range(4):
            for j in range(5):
                pixel_rst_time.append(self.pixels[i][j].rst_time)
        self.rst_time = max(pixel_rst_time)

        self.readout_time = self.compute_readout_time(self.max_output_voltage, self.min_output_voltage)
        if Pixel_type == 0:
            self.fd_reset_time = max(self.pixels[0][0].fd_rst_time, self.readout_time)
        else:
            self.fd_reset_time = 0

        self.sf_max_input = self.pixels[0][0].rst_voltage
        self.sf_min_input = self.compute_sf_input_voltage(self.min_output_voltage)
        pixels_exposure_time = []
        for i in range(4):
            for j in range(5):
                temp = self.pixels[i][j].compute_exposure_time(self.pixels[i][j].rst_voltage - self.sf_min_input)
                pixels_exposure_time.append(temp)
        self.max_exposure_time = min(pixels_exposure_time)
        self.exposure_time = exposure_time if exposure_time != 0 else self.max_exposure_time

        # CNN-specific timing
        if Pixel_type == 2:
            self.pixel_time = self.pixels[0][0].compute_exposure_time(self.pixels[0][0].rst_voltage - self.sf_min_input) + self.rst_time + self.fd_reset_time
            if self.pixel_time > 1/input_clk_freq*(2**4):
                self.pixel_readout_time = self.pixel_time * 2*CNN_kernel*CNN_kernel
            else:
                self.pixel_readout_time = 1/input_clk_freq*(2**4) * 2*CNN_kernel*CNN_kernel
        else:
            # time need for a PMW comparator to generate the ramp can count for 4bit
            self.pixel_readout_time = 1/input_clk_freq*(2**4) * 2*CNN_kernel*CNN_kernel
        
        self.readout_circuit_time = input_clk_freq*4 + 1/input_clk_freq*(2**4) + input_clk_freq*8
        self.frame_time = self.exposure_time + self.pixel_readout_time + self.readout_circuit_time
        self.max_frame_rate = 1/self.frame_time
        self.frame_rate = frame_rate if frame_rate > 1 and frame_rate <= self.max_frame_rate else self.max_frame_rate

        # Power calculations
        self.compute_power(num_rows, num_cols, Pixel_type, frame_rate, input_clk_freq, load_cap, 
                          pixel_binning_map, CNN_kernel)

        # Noise calculations
        self.compute_noise(num_rows, num_cols, Pixel_type, adc_resolution, num_PD_per_tap, 
                          pixel_binning_map, CNN_kernel)

        # Print results
        if self.print_output:
            self.print_timing_results(frame_rate)
            self.print_power_consumption(Pixel_type)
            self.print_noise_results()

    def compute_power(self, num_rows, num_cols, Pixel_type, frame_rate, input_clk_freq, load_cap,
                     pixel_binning_map, CNN_kernel):
        """Compute CNN-specific power consumption"""
        # Reset power
        if Pixel_type == 0:
            self.rst_power = self.compute_rst_energy(
                self.pixels[0][0].fd_cap + self.pixels[0][0].pd_capacitance + self.pixels[0][0].tg.source_cap
            ) * frame_rate * num_cols * num_rows
        elif Pixel_type == 1:
            self.rst_power = self.compute_rst_energy(
                self.pixels[0][0].Cap_FB + self.pixels[0][0].load_cap
            ) * frame_rate * num_cols * num_rows
        elif Pixel_type == 2:
            self.rst_power = self.compute_rst_energy(
                CNN_kernel**2 * self.pixels[0][0].fd_cap + self.pixels[0][0].pd_capacitance + self.pixels[0][0].tg.source_cap
            ) * frame_rate * num_cols * num_rows
        else:
            self.rst_power = self.compute_rst_energy(
                self.pixels[0][0].pd_capacitance + self.pixels[0][0].tg.source_cap
            ) * frame_rate * num_cols * num_rows

        # Pixel power
        if Pixel_type == 2:
            num_cols_cnn = num_cols/CNN_kernel
            num_rows_cnn = num_rows/CNN_kernel
            self.pixel_power = self.rst_power + self.PMW_comparator.power * 1/input_clk_freq*(2**4) * num_cols_cnn*num_cols_cnn*frame_rate/(CNN_kernel**2)
            self.pixel_power += self.TIA.power
        else:
            self.pixel_power = self.rst_power + self.PMW_comparator.power * 1/input_clk_freq*(2**4) * num_cols*num_cols*frame_rate

        # Ramp generator power
        self.ramp_generator_power = (
            (self.PMW_ramp_generator.charge_energy + self.PMW_ramp_generator.discharge_energy) * 
            frame_rate * 2 * CNN_kernel * CNN_kernel * num_rows / (2*CNN_kernel) + 
            (self.readout_ramp_generator.charge_energy + self.readout_ramp_generator.discharge_energy) * 
            frame_rate * num_rows / (2*CNN_kernel)
        )

        # CNN logic power
        self.polarity_judge_power = 1/2 * self.inv.total_switch_cap * (self.analog_V_dd**2) * 2 * frame_rate * num_rows*num_cols/(CNN_kernel**2)
        self.CNN_logic_cap_power = 1/2 * load_cap * (self.analog_V_dd**2) * 4 * frame_rate * num_cols * num_rows/(CNN_kernel**2)
        self.CNN_logic_switch_power = 1/2 * self.bias_transistor.gate_cap * (self.analog_V_dd**2) * (14+(2*2)) * frame_rate * num_rows * num_cols/(CNN_kernel**2)
        self.CNN_logic_comparator_power = self.comparator.power * num_cols/CNN_kernel
        self.CNN_logic_power = self.polarity_judge_power + self.CNN_logic_cap_power + self.CNN_logic_switch_power + self.CNN_logic_comparator_power

        # Readout circuit power
        self.or_gate_power = (self.nor.total_switch_cap + self.inv.total_switch_cap) * (self.analog_V_dd**2) * frame_rate * num_cols*num_rows/(4*(CNN_kernel**2))
        self.and_gate_power = (self.nand.total_switch_cap + self.inv.total_switch_cap) * (self.analog_V_dd**2) * frame_rate * 4 * num_rows*num_cols/(4*(CNN_kernel**2))
        self.counter_power = self.CNN_case_counter.total_power
        self.B2T_power = 1/2 * (self.nand.total_switch_cap + 5*self.inv.total_switch_cap + 3*self.nor.total_switch_cap) * (self.analog_V_dd**2) * frame_rate * 4 * num_cols*num_rows/(4*(CNN_kernel**2))
        self.up_down_counter_power = self.up_down_counter.total_power
        self.readout_total_power = self.or_gate_power + self.and_gate_power + self.counter_power + self.B2T_power + self.up_down_counter_power

        # Bus power
        self.bus_energy_per_transition_horizontal = self.buffer.unit_energy * self.horizontal_bus_length
        self.analog_bus_energy_per_transition_horizontal = self.Analog_Buffer.unit_energy * self.horizontal_bus_length
        self.CNN_logic_bus_power = self.bus_energy_per_transition_horizontal * 10 * num_rows/3
        self.Readout_bus_power = self.bus_energy_per_transition_horizontal * 3
        self.Analog_ramp_bus_power = self.analog_bus_energy_per_transition_horizontal * 2  # two ramp generator
        self.bus_power = self.CNN_logic_bus_power + self.Readout_bus_power
        self.digital_bus_power = self.CNN_logic_bus_power + self.Readout_bus_power
        self.analog_bus_power = self.Analog_ramp_bus_power

        # System total power
        self.system_total_power = self.pixel_power + self.ramp_generator_power + self.CNN_logic_power + self.readout_total_power + self.bus_power

    def compute_noise(self, num_rows, num_cols, Pixel_type, adc_resolution, num_PD_per_tap, 
                     pixel_binning_map, CNN_kernel):
        """Compute noise for CNN CIS"""
        for i in range(4):
            for j in range(5):
                self.pixels[i][j].bias_current = self.bias_current

        # ADC input cap
        if Pixel_type == 2:
            self.ADC_input_cap = CNN_kernel**2 * self.pixels[0][0].fd_cap + self.pixels[0][0].pd_capacitance + self.pixels[0][0].tg.source_cap
        elif Pixel_type == 1:
            self.ADC_input_cap = self.pixels[0][0].Cap_FB
        else:
            self.ADC_input_cap = self.pixels[0][0].fd_cap

        self.ADC_Noise_model = ADC_Quantizing_Noise(
            ADC_resolution=adc_resolution,
            voltage_range=self.digital_V_dd - self.min_output_voltage,
            cap=self.ADC_input_cap
        )

        # Reset noise
        if Pixel_type == 0:
            self.rst_cap = self.pixels[0][0].fd_cap + self.pixels[0][0].pd_capacitance + self.pixels[0][0].tg.source_cap
        elif Pixel_type == 1:
            self.rst_cap = self.pixels[0][0].Cap_FB + self.pixels[0][0].load_cap
        elif Pixel_type == 2:
            self.rst_cap = CNN_kernel**2 * self.pixels[0][0].fd_cap + self.pixels[0][0].pd_capacitance + self.pixels[0][0].tg.source_cap
        else:
            self.rst_cap = self.pixels[0][0].pd_capacitance + self.pixels[0][0].tg.source_cap

        self.Reset_Noise_model = Reset_Noise(cap=self.rst_cap, temperature=300)
        self.reset_noise = self.Reset_Noise_model.reset_noise

        # Noise calculations
        self.ADC_quantization_noise_square = self.ADC_Noise_model.quant_noise_square
        for i in range(4):
            for j in range(5):
                self.pixels[i][j].sf_vds = self.analog_V_dd - 0.5*(self.max_output_voltage + self.min_output_voltage)
                self.pixels[i][j].load_cap = self.load_cap

        self.photo_shot_noise_square = self.pixels[0][0].shot_noise_model.shot_noise_square
        self.sf_shot_noise_square = self.pixels[0][0].current_shot_noise_model.current_shot_noise_square
        self.sf_thermal_noise_square = self.pixels[0][0].sf_thermal_noise_model.thermal_noise_square
        self.tg_thermal_noise_square = self.pixels[0][0].tg_thermal_noise_model.thermal_noise_square
        self.total_thermal_noise_square = self.sf_thermal_noise_square + self.tg_thermal_noise_square
        self.dark_current_noise_square = self.pixels[0][0].dark_current_noise_model.dark_current_noise_square
        self.transfer_noise_square = self.pixels[0][0].transfer_noise_model.transfer_noise_square
        self.total_shot_noise_square = (self.photo_shot_noise_square + self.sf_shot_noise_square + self.dark_current_noise_square) * pixel_binning_map[0]*pixel_binning_map[1]
        self.total_read_noise_square = self.total_thermal_noise_square + self.ADC_quantization_noise_square
        self.total_noise_square = self.total_shot_noise_square + self.total_read_noise_square + self.transfer_noise_square

        # SNR and DR calculations
        self.SNR = []
        self.DR = []
        self.FWC = []
        for i in range(4):
            self.local_SNR = []
            self.local_DR = []
            self.local_FWC = []
            for j in range(5):
                photo_shot_noise_square = (self.pixels[i][j].current_shot_noise_model.current_shot_noise_square + 
                                          self.pixels[i][j].current_shot_noise_model.current_shot_noise_square + 
                                          self.pixels[i][0].dark_current_noise_model.dark_current_noise_square)
                self.signal = self.pixels[i][j].shot_noise_model.shot_noise_square * pixel_binning_map[0] * pixel_binning_map[1]
                self.SNR_noise_square = self.pixels[i][j].shot_noise_model.shot_noise_square + self.total_read_noise_square + self.transfer_noise_square
                self.local_SNR.append(20*math.log10(self.signal/math.sqrt(self.SNR_noise_square)))
                self.FWC_temp = self.pixels[i][j].PD.FWC * num_PD_per_tap * pixel_binning_map[0] * pixel_binning_map[1]
                self.local_FWC.append(self.pixels[i][j].PD.FWC * num_PD_per_tap * pixel_binning_map[0] * pixel_binning_map[1])
                self.local_DR.append(20*math.log10(self.FWC_temp/math.sqrt(self.total_read_noise_square)))
            self.SNR.append(self.local_SNR)
            self.DR.append(self.local_DR)
            self.FWC.append(self.local_FWC)

    def compute_bias_current(self):
        bias_current = 0.5 * self.bias_transistor.un * self.bias_transistor.Cox * (self.bias_transistor.width/self.bias_transistor.length) * ((self.bias_voltage - self.bias_transistor.v_th) **2) * (1+self.bias_transistor.lambda_param*self.bias_voltage)
        return bias_current

    def compute_readout_time(self, max_voltage, min_voltage):
        sf_output_resistance = 1/math.sqrt(2 * self.pixels[0][0].sf.un * self.pixels[0][0].sf.Cox * (self.pixels[0][0].sf.width / self.pixels[0][0].sf.length) * self.bias_current)
        bias_transistor_resistance = 1/(self.bias_transistor.lambda_param*self.bias_current)
        output_resistance = (sf_output_resistance * bias_transistor_resistance) / (sf_output_resistance + bias_transistor_resistance)
        total_capacitance = self.pixels[0][0].sf.source_cap + self.output_bus.cap_wire_per_m*self.output_bus_length + self.load_cap + self.PGA.input_cap + self.PGA.comparator.input_transistor.gate_cap
        total_resistance = output_resistance + self.output_bus.res_wire_per_m*self.output_bus_length
        t_slew = (max_voltage - min_voltage) * total_capacitance / self.bias_current
        t_settle = total_resistance * total_capacitance * 5
        readout_time = t_slew + t_settle
        return readout_time

    def compute_sf_input_voltage(self, sf_output_voltage):
        Vgs = math.sqrt((2*self.bias_current) / (self.pixels[0][0].sf.un * self.pixels[0][0].sf.Cox * self.pixels[0][0].sf.width / self.pixels[0][0].sf.length)) + self.pixels[0][0].sf.v_th
        sf_input_voltage = sf_output_voltage + Vgs
        return sf_input_voltage

    def compute_sf_out_voltage(self, sf_input_voltage):
        Vgs = math.sqrt((2*self.bias_current) / (self.pixels[0][0].sf.un * self.pixels[0][0].sf.Cox * self.pixels[0][0].sf.width / self.pixels[0][0].sf.length)) + self.pixels[0][0].sf.v_th
        sf_output_voltage = sf_input_voltage - Vgs
        return sf_output_voltage

    def compute_rc_lpf(self, reference_freq_hz, divider=10):
        fc = reference_freq_hz / divider
        rc = 1 / (2 * math.pi * fc)
        R = 1e3  # Ohms
        C = rc / R  # Farads
        return R, C

    def compute_rst_energy(self, total_cap):
        energy = total_cap * self.pixels[0][0].rst_voltage * self.analog_V_dd
        return energy

    def format_time(self, value):
        if value >= 1e-3:
            return f"{value * 1e3:10.2f} ms"
        else:
            return f"{value * 1e6:10.2f} µs"

    def format_power(self, value):
        if value >= 1e-3:
            return f"{value * 1e3:10.2f} mW"
        else:
            return f"{value * 1e6:10.2f} µW"

    def print_timing_results(self, frame_rate):
        """Print timing model results for CNN CIS"""
        print(f"========= TIMING MODEL RESULTS =========")
        print(f"Frame Time : {self.format_time(self.frame_time)}")
        print(f"  └─ Exposure Time : {self.format_time(self.exposure_time)}")
        print(f"  └─ Pixel Readout Time : {self.format_time(self.pixel_readout_time)}")
        print(f"  └─ CNN Processing Time : {self.format_time(self.readout_circuit_time)}")
        print(f"Frame Rate : {frame_rate} Hz")
        print(f"  └─ Max Frame Rate : {1/self.frame_time} Hz")
        print("-" * 40)

    def print_power_consumption(self, Pixel_type):
        """Print power consumption results for CNN CIS"""
        print(f"========= POWER CONSUMPTION (W) ========")
        print(f"System Total Power : {self.format_power(self.system_total_power)}")
        print("-" * 40)
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
        print("========================================")

    def print_noise_results(self):
        """Print noise analysis results for CNN CIS"""
        print("========= Noise =========")
        print(f"Dynamic range")
        for i in range(4):
            for j in range(5):
                print(f"    {i}th Color, {j}th pixel size  : {self.DR[i][j]:.2f} dB")
        print(f"Signal Noise Ratio")
        for i in range(4):
            for j in range(5):
                print(f"    {i}th Color, {j}th pixel size  : {self.SNR[i][j]:.2f} dB")
        for i in range(4):
            for j in range(5):
                print(f"    {i}th Color, {j}th pixel size  : {self.FWC[i][j]} e-")
        print(f"\nTotal Noise      : {math.sqrt(self.total_noise_square):.2f} e-")
        print(f"  1. Total Shot Noise     : {math.sqrt(self.total_shot_noise_square):.2f} e-")
        print(f"    ├─ Photo Shot Noise       : {math.sqrt(self.photo_shot_noise_square):.2f} e-")
        print(f"    ├─ SF Shot Noise          : {math.sqrt(self.sf_shot_noise_square):.2f} e-")
        print(f"    └─ Dark Current Shot Noise: {math.sqrt(self.dark_current_noise_square):.2f} e-")
        print(f"  2. Total Read Noise     : {math.sqrt(self.total_read_noise_square):.2f} e-")
        print(f"    ─ Total Thermal Noise     : {math.sqrt(self.total_thermal_noise_square *1e6):.2f} µe-")
        print(f"       ├─ SF Thermal Noise    : {math.sqrt(self.sf_thermal_noise_square*1e6):.2f} µe-")
        print(f"       ├─ TG Thermal Noise    : {math.sqrt(self.tg_thermal_noise_square*1e6):.2f} µe-")
        print(f"    ─ ADC Quantization Noise  : {math.sqrt(self.ADC_quantization_noise_square*1e6):.2f} µe-")
        print(f"  3. Transfer(CTI) Noise  : {math.sqrt(self.transfer_noise_square):.2f} e-")

