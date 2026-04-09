import numpy as np
from APS import FourTAPS
from wire import Wire
from parameter_class import NMOS
#from SH import SHCircuit
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
from SRAM_Buffer import SRAM
from Noise import Shot_Noise, Dark_Current_Noise, Thermal_Noise, RTS_Noise, Transfer_Noise, Current_Shot_Noise, ADC_Quantizing_Noise, System_Noise, Reset_Noise
from SS_ADC import SS_ADC
from SAR_ADC import SAR_ADC
from CDS import CDS

# APS Array class
# Models a complete active pixel sensor array with pixels, readout circuits, ADC, PLL, and buffers
# 
# Inputs: pixel_type (pixel type string, e.g., "4T_APS"), num_rows (number of rows), num_cols (number of columns),
#         pd_eta (quantum efficiency), pd_E (optical power density in W/m²), shutter (shutter type: "rolling" or "global"),
#         input_clk_freq (input clock frequency in Hz), feature_size_nm (process technology node in nm, optional, default=65),
#         pd_length (photodiode length in μm, optional, default=10), pd_width (photodiode width in μm, optional, default=10),
#         light_source (light source type, optional, default=1), V_dd (supply voltage in V, optional, default=1.8),
#         temperature (temperature in K, optional, default=293), bias_voltage (bias voltage in V, optional, default=0.6),
#         load_cap (load capacitance in F, optional, default=5e-12), ramp_bias_voltage (ramp bias voltage in V, optional, default=0.6),
#         ramp_load_cap (ramp load capacitance in F, optional, default=5e-12), comparator_bias_voltage (comparator bias voltage in V, optional, default=0.8),
#         comparator_load_cap (comparator load capacitance in F, optional, default=5e-12), comparator_input_cap (comparator input capacitance in F, optional, default=5e-12),
#         charge_pump_load_cap (charge pump load capacitance in F, optional, default=5e-12), V_ctrl (control voltage in V, optional, default=0.6),
#         frame_rate (frame rate in fps, optional, default=0), adc_resolution (ADC resolution in bits, optional, default=12),
#         buffer_flag (buffer enable flag, optional, default=1), MIPI_unit_cap (MIPI unit capacitance in F, optional, default=0),
#         FWC (full well capacity in electrons, optional, default=0), V_swing (voltage swing in V, optional, default=0.4),
#         ADC_type (ADC type: 0=SS_ADC, 1=SAR_ADC, optional, default=0), system_clk_freq (system clock frequency in Hz, optional, default=0),
#         CDS_amp_gain (CDS amplifier gain, optional, default=5)
# Outputs: frame_time (frame time in s), frame_rate (frame rate in fps), total_power (total system power in W),
#          pixel_power (pixel array power in W), adc_power (ADC power in W), pll_power (PLL power in W),
#          total_bus_power (total bus power in W), input_driver_power (input driver power in W), MIPI_power (MIPI power in W),
#          SRAM_buffer_power (SRAM buffer power in W), and various noise and timing metrics (SNR, DR, noise components, etc.)
class APSArray:
    def __init__(self, pixel_type, num_rows, num_cols, pd_eta, pd_E, shutter, input_clk_freq, feature_size_nm = 65,
                  pd_length = 10, pd_width = 10, light_source = 1, V_dd = 1.8, temperature = 293,
                  bias_voltage = 0.6, load_cap = 5e-12, ramp_bias_voltage = 0.6, ramp_load_cap = 5e-12,
                  comparator_bias_voltage = 0.8, comparator_load_cap = 5e-12, comparator_input_cap = 5e-12, 
                  charge_pump_load_cap = 5e-12, V_ctrl = 0.6, frame_rate = 0, adc_resolution = 12, buffer_flag = 1,
                  MIPI_unit_cap = 0, FWC = 0, V_swing = 0.4, ADC_type = 0, system_clk_freq = 0, CDS_amp_gain = 5):
        self.Q = 1.602e-19
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.temperature = temperature
        self.shutter = shutter #TODO add golbal shutter
        self.V_dd = V_dd
        self.bias_voltage = bias_voltage
        self.load_cap = load_cap
        self.comparator_bias_voltage = comparator_bias_voltage
        self.comparator_load_cap = comparator_load_cap
        self.comparator_input_cap = comparator_input_cap
        self.ramp_bias_voltage = ramp_bias_voltage
        self.ramp_load_cap = ramp_load_cap
        self.charge_pump_load_cap = charge_pump_load_cap
        self.ADC_resolution = adc_resolution #by default
        self.counter_clk_freq = 0
        self.V_ctrl = V_ctrl
        self.buffer_flag = buffer_flag
        self.PD_E = pd_E
        self.FF = 0.35

        self.bias_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.bias_current = self.compute_bias_current()

        self.pixel = FourTAPS(
            pd_eta=pd_eta,
            pd_E=pd_E,
            feature_size_nm=feature_size_nm,
            pd_length=pd_length,
            pd_width= pd_width,
            light_source=light_source,
            bias_current=self.bias_current,
            V_dd=V_dd,
            FWC = FWC,
            V_swing = V_swing
        )


        if num_cols > 100 : #TODO, find a proper value determine the wire type
            self.input_bus_wire_type = "global_aggressive"
        else:
            self.input_bus_wire_type = "semi_aggressive"
        
        if num_rows > 100:
            self.output_bus_wire_type = "global_aggressive"
        else:
            self.output_bus_wire_type = "semi_aggressive"

        self.input_bus_length = num_cols * 1e-6 
        self.output_bus_length = num_rows * 1e-6 

        self.input_bus = Wire(
            wire_type=self.input_bus_wire_type,
            feature_size_nm=feature_size_nm,
            temperature=self.temperature
        )

        self.output_bus = Wire(
            wire_type=self.output_bus_wire_type,
            feature_size_nm=feature_size_nm,
            temperature=self.temperature
        )
        self.rst_time = self.pixel.rst_time

        self.max_output_voltage = self.compute_sf_out_voltage(self.pixel.rst_voltage)
        self.min_output_voltage = self.bias_voltage - self.bias_transistor.v_th # the min output voltage to maintain the bias transistor in Saturation
        self.sf_max_input = self.pixel.rst_voltage
        self.sf_min_input = self.compute_sf_input_voltage(self.min_output_voltage) #the min input voltage to maintain the bias transistor in Saturation

        self.exposure_time = self.pixel.compute_exposure_time(self.pixel.rst_voltage - self.sf_min_input)
        self.input_delay, _ = self.compute_input_bus_delay_energy()


        self.buffer = Repeater(
            feature_size_nm=feature_size_nm,
            capWirePerUnit=self.input_bus.cap_wire_per_m,
            resWirePerUnit=self.input_bus.res_wire_per_m,
            V_dd=self.V_dd
        )
        #compute the total cap between the SF and the comparator
        
        self.SS_ADC = SS_ADC(
            ramp_bias_voltage = ramp_bias_voltage, ramp_load_cap = ramp_load_cap, 
            max_output_voltage = self.max_output_voltage, adc_resolution = adc_resolution,
            comparator_bias_voltage = comparator_bias_voltage, comparator_load_cap = comparator_load_cap, comparator_input_cap = comparator_input_cap, 
            num_cols = num_cols, num_rows = num_rows, feature_size_nm = feature_size_nm, V_dd=V_dd
        )

        self.counter_clk_freq  = (2**self.ADC_resolution)/self.SS_ADC.ramp_generator.ADC_time

        if system_clk_freq:
            if system_clk_freq > self.counter_clk_freq:
                print("reference system CLK freq is larger than max CLK req")
            else:
                self.counter_clk_freq = system_clk_freq

        self.CDS = CDS (V_rst = self.max_output_voltage, V_sig = self.min_output_voltage, comaprator_FB_cap = CDS_amp_gain*comparator_input_cap,
            comparator_bias_voltage = comparator_bias_voltage, comparator_load_cap = comparator_load_cap, comparator_input_cap = comparator_input_cap, 
            num_cols = num_cols, num_rows = num_rows, frame_rate = frame_rate,
            feature_size_nm = feature_size_nm, V_dd=V_dd, DC_gain = 2000
        )
        
        self.SAR_ADC = SAR_ADC(
            max_output_voltage = self.CDS.CDS_output_voltage, adc_resolution = adc_resolution,
            comparator_bias_voltage = comparator_bias_voltage, comparator_load_cap = comparator_load_cap, comparator_input_cap = comparator_input_cap, 
            num_cols = num_cols, num_rows = num_rows, frame_rate = frame_rate,
            input_clk_freq = self.counter_clk_freq, feature_size_nm = feature_size_nm, V_dd=V_dd, DC_gain = 2000
        )

        self.readout_time = self.compute_readout_time(self.max_output_voltage, self.min_output_voltage)
        self.fd_reset_time = max(self.pixel.fd_rst_time, self.readout_time)


        #Timing Model Final result, the fram time 
        if ADC_type == 0:
            self.sel_time = self.fd_reset_time + self.SS_ADC.comparator.AZ_time + self.SS_ADC.ramp_generator.CDS_time + self.SS_ADC.ramp_generator.ADC_time
            + self.readout_time + self.SS_ADC.ramp_generator.reset_time
        else:
            self.sel_time = self.fd_reset_time + self.readout_time + self.SAR_ADC.delay + self.CDS.CDS_delay
        self.frame_time = max(self.sel_time*num_rows, self.rst_time+self.exposure_time+self.sel_time)
        #if add buffer in the bus
        if buffer_flag:
            self.bus_delay = self.buffer.unit_delay*self.input_bus_length
        else:
            self.bus_delay,_ = self.compute_input_bus_delay_energy()
        #TODO: readout delay
        self.critical_path = self.rst_time+self.exposure_time+self.sel_time+self.bus_delay
        self.frame_rate = 1/self.frame_time
        
        #if user predefine a frame rate
        if frame_rate > 1:
            if frame_rate > self.frame_rate:
                print("Input frame rate is too low, will use largest frame rate calcualted by the model")
                self.frame_rate = self.frame_rate
            else:
                self.frame_rate = frame_rate

        #Update the ADC frame rate to calcualte the energy
        self.SS_ADC.frame_rate = self.frame_rate
        self.SAR_ADC.frame_rate = self.frame_rate

        #if user has input, use user's input. else use calculated max clk freq
        if(input_clk_freq):
            self.input_clk_freq = input_clk_freq
        else:
            self.input_clk_freq = self.counter_clk_freq


        self.pi = 3.1415926

        #Here the divider means pass clk freq/divider frequeny
        self.LPF_resistance,self.LPF_load_cap = self.compute_rc_lpf(self.input_clk_freq, divider=10)

        self.PLL = Phase_lock_loop(
            CP_load_cap = self.charge_pump_load_cap, 
            LPF_resistance = self.LPF_resistance,
            LPF_load_cap = self.LPF_load_cap, 
            input_clk_freq = self.input_clk_freq, 
            output_clk_frequency = self.counter_clk_freq, 
            bias_votlage = self.bias_voltage, 
            feature_size_nm = feature_size_nm, 
            V_dd=self.V_dd,
            V_ctrl=self.V_ctrl
        )

        self.old_frame_time = self.frame_time

        #Ideal driver input clk freq
        self.driver_input_clk_freq = 1/(self.frame_time/self.num_rows)
        N = self.counter_clk_freq / self.driver_input_clk_freq
    
        # Find the closest power of 2
        k_closest = max(1, int(np.ceil(np.log2(N))))
        N_closest = 2 ** k_closest

        # Calculate the actuall driver input frequency
        self.driver_input_clk_freq = self.counter_clk_freq / N_closest
        self.frame_time = (1/self.driver_input_clk_freq)*self.num_rows

        #Update frame rate after the PLL was added
        self.idle_time = self.frame_time - self.old_frame_time
        self.frame_rate = 1/self.frame_time

        self.Input_Driver = Input_Driver(
            input_clk_freq=self.driver_input_clk_freq,
            num_rows=self.num_rows,
            feature_size_nm=feature_size_nm,
            V_dd=self.V_dd
        )

        self.Analog_Buffer = Analog_Buffer_Bus(
            wire_unit_cap=self.input_bus.cap_wire_per_m,
            wire_unit_res=self.input_bus.res_wire_per_m,
            feature_size_nm=feature_size_nm,
            V_dd=self.V_dd,
            bias_voltage=self.bias_voltage
        )

        self.MIPI_unit_cap = MIPI_unit_cap

        self.MIPI = MIPI(
            data_bandwidth =self.ADC_resolution*self.num_cols, 
            frame_rate = self.frame_rate, 
            MIPI_unit_cap = self.MIPI_unit_cap, 
            feature_size_nm = feature_size_nm, 
            MIPI_unit_energy = 100e-12
        )

        self.SRAM_buffer = SRAM(
            frame_rate = self.frame_rate,
            next_stage_input_cap = self.MIPI.MIPI_cap,
            input_clk_freq = self.driver_input_clk_freq, 
            num_cols = self.num_cols, 
            size = self.ADC_resolution*self.num_cols*2, 
            feature_size_nm = feature_size_nm,
            V_dd = self.V_dd
        )

       

        #total energy consumption per pixel per frame
        self.pixel_energy = self.compute_rst_readout_energy() + self.compute_rst_energy() + self.compute_signal_readout_energy()
        self.rst_power = self.compute_rst_energy()*self.frame_rate*self.num_cols*self.num_rows
        self.sf_power = (self.compute_rst_readout_energy()+self.compute_signal_readout_energy())*self.frame_rate*self.num_cols*self.num_rows
        # print("readout energy", self.compute_signal_readout_energy())
        # print("bias current", self.bias_current)
        self.pixel_power = self.rst_power+self.sf_power


        #energy consumption per adc per operation (number of rows operations per frame)
        self.adc_power = self.SS_ADC.adc_power
        self.pll_power = self.PLL.total_dynamic_power
        self.PFD_power = self.PLL.PFD_dynamic_power
        self.CP_power = self.PLL.CP_dynamic_power
        self.VCO_power = self.PLL.VCO_dynamic_power
        self.FD_power = self.PLL.FD_dynamic_power

        #energy consumption per transition per bus.
        #there are 3bus per row, where TG and RST buses toggle twice per frame, SEL buses toggle once per frame
        if buffer_flag:
            self.bus_energy_per_transition = self.buffer.unit_energy*self.input_bus_length
            self.analog_bus_energy_per_transition = self.Analog_Buffer.unit_energy*self.input_bus_length
        else:
            _,self.bus_energy_per_transition = self.compute_input_bus_delay_energy()
            _,self.analog_bus_energy_per_transition = self.compute_input_bus_delay_energy()

        #SEL and RST bus 2 times transitions(rise and fall) per frame TG 1 transition per frame
        self.pixel_bus_power = ((self.bus_energy_per_transition*2*self.frame_rate)*2 + (self.bus_energy_per_transition*self.frame_rate))*self.num_rows
        #AZ 1 transition per frame
        self.az_bus_power = (self.bus_energy_per_transition*self.frame_rate)*self.num_rows
        #Clock 1 transition per clk
        self.clk_bus_power = self.bus_energy_per_transition*self.counter_clk_freq
        #Analog(ramp generator); x2 because the ramp generator rise and fall twice during one readout
        self.pixel_output_bus_power = self.Analog_Buffer.unit_energy*self.output_bus_length*self.frame_rate*self.num_cols
        self.analog_bus_power = self.analog_bus_energy_per_transition*self.frame_rate*self.num_rows*2 + self.pixel_output_bus_power
        #TODO:clk bus power to the input driver
        self.total_bus_power = self.pixel_bus_power + self.az_bus_power + self.clk_bus_power + self.analog_bus_power
        
        #Input driver power
        self.input_driver_power_rst = self.Input_Driver.total_power * 2
        self.input_driver_power_tg = self.Input_Driver.total_power * 2
        self.input_driver_power_sel = self.Input_Driver.total_power
        self.input_driver_power = self.input_driver_power_rst + self.input_driver_power_tg + self.input_driver_power_sel

        #Data Transimission Power
        self.MIPI_power = self.MIPI.MIPI_power

        #SRAM Buffer Power
        #only consider the SRAM cells read and write power
        self.SRAM_buffer_power = self.SRAM_buffer.total_power
        self.SRAM_read_power = self.SRAM_buffer.read_power
        self.SRAM_write_power = self.SRAM_buffer.write_power
        self.SRAM_bus_power = self.SRAM_buffer.bus_power
        self.SRAM_counter_power = self.SRAM_buffer.counter_power
        self.SRAM_mux_power = self.SRAM_buffer.mux_power
        self.SRAM_decoder_power = self.SRAM_buffer.decoder_power

        self.total_power = self.pixel_power+self.adc_power+self.pll_power+self.total_bus_power+self.input_driver_power+self.MIPI_power+self.SRAM_buffer_power
        
        ################################################ NOISE ###################################################
        self.pixel.bias_current = self.bias_current
        self.ADC_input_cap = self.load_cap + self.pixel.sf.drain_cap + self.bias_transistor.drain_cap + self.comparator.input_cap
        self.ADC_Noise_model = ADC_Quantizing_Noise(
            ADC_resolution=self.ADC_resolution,
            voltage_range=self.max_output_voltage-self.min_output_voltage,
            cap=self.ADC_input_cap
        )

        self.ADC_thermal_noise_model = Thermal_Noise(
            transconductance=self.SS_ADC.comparator.Gm,
            cap=self.ADC_input_cap,
            temperature=300
        )

        self.ADC_current_shot_noise_model = Current_Shot_Noise(
            current=self.SS_ADC.comparator.bias_current,
            transconductance=self.SS_ADC.comparator.Gm,
            input_cap=self.ADC_input_cap,
            load_cap=self.SS_ADC.comparator.output_cap + self.SS_ADC.counter.input_cap,
            output_res=self.SS_ADC.comparator.output_res
        )

        self.Reset_Noise_model = Reset_Noise(
            cap=self.pixel.fd_cap+self.pixel.pd_capacitance+self.pixel.tg.source_cap,
            temperature=temperature
        )

        self.reset_noise = self.Reset_Noise_model.reset_noise
        self.ADC_quantization_noise_square = self.ADC_Noise_model.quant_noise_square
        self.ADC_thermal_noise_square = self.ADC_thermal_noise_model.thermal_noise_square
        self.ADC_current_shot_noise_square = self.ADC_current_shot_noise_model.current_shot_noise_square

        self.pixel.sf_vds = self.V_dd-0.5*(self.max_output_voltage+self.min_output_voltage)
        self.pixel.load_cap = self.load_cap

        self.photo_shot_noise_square = self.pixel.shot_noise_model.shot_noise_square
        self.sf_shot_noise_square = self.pixel.current_shot_noise_model.current_shot_noise_square
        self.sf_thermal_noise_square = self.pixel.sf_thermal_noise_model.thermal_noise_square
        self.tg_thermal_noise_square = self.pixel.tg_thermal_noise_model.thermal_noise_square

        self.total_thermal_noise_square = self.sf_thermal_noise_square + self.tg_thermal_noise_square + self.ADC_thermal_noise_square
        self.dark_current_noise_square = self.pixel.dark_current_noise_model.dark_current_noise_square
        self.transfer_noise_square = self.pixel.transfer_noise_model.transfer_noise_square
        self.total_shot_noise_square = self.ADC_current_shot_noise_square + self.photo_shot_noise_square + self.sf_shot_noise_square+self.dark_current_noise_square
        self.total_read_noise_square = self.total_thermal_noise_square+self.ADC_quantization_noise_square

        self.total_noise_square = self.total_shot_noise_square+self.total_read_noise_square+self.transfer_noise_square
        # self.SNR = 20*math.log10(self.photo_shot_noise_square/math.sqrt(self.total_noise_square))
    
        # self.DR = 20*math.log10(FWC/math.sqrt(self.total_noise_square))

    ### Physical Modeling
        if ADC_type == 0:
            #TG, RST, SEL, AZ, CLK
            self.bus_length = 5 * self.input_bus_length
            #Ramp input
            self.analog_bus_length = self.input_bus_length
        else:
            #TG, RST, SEL, CLK, START_CONVERSION
            self.bus_length = 5 * self.input_bus_length
            #Vref
            self.analog_bus_length = self.input_bus_length

    def compute_sf_out_voltage(self, sf_input_voltage) :
        Vgs = math.sqrt((2*self.bias_current) / (self.pixel.sf.un * self.pixel.sf.Cox * self.pixel.sf.width / self.pixel.sf.length)) + self.pixel.sf.v_th
        sf_output_voltage = sf_input_voltage - Vgs
        return sf_output_voltage

    def compute_sf_input_voltage(self, sf_output_voltage):
        Vgs = math.sqrt((2*self.bias_current) / (self.pixel.sf.un * self.pixel.sf.Cox * self.pixel.sf.width / self.pixel.sf.length)) + self.pixel.sf.v_th
        sf_input_voltage = sf_output_voltage + Vgs
        return sf_input_voltage


    def compute_bias_current(self):
        bias_current = 0.5 * self.bias_transistor.un * self.bias_transistor.Cox * (self.bias_transistor.width/self.bias_transistor.length) * ((self.bias_voltage - self.bias_transistor.v_th) **2)*(1+self.bias_transistor.lambda_param*self.bias_voltage)
        return bias_current

    # def compute_output_voltage(self):
    #     output_voltage = self.V_dd - self.pixel.rst_transistor.v_th - self.bias_transistor.v_th
    #     return output_voltage

    def compute_readout_time(self, max_voltage, min_voltage):
        #compute the readout time
        #max_voltage: max possible voltage of the output cap
        #min_voltage: min possible voltage of the output cap
        #TODO: add gmb
        sf_gm = math.sqrt(2 * self.pixel.sf.un * self.pixel.sf.Cox * (self.pixel.sf.width / self.pixel.sf.length) * self.bias_current)
        
        sf_output_resistance = 1/math.sqrt(2 * self.pixel.sf.un * self.pixel.sf.Cox * (self.pixel.sf.width / self.pixel.sf.length) * self.bias_current) #1/gm 
        bias_transistor_resistance = 1/(self.bias_transistor.lambda_param*self.bias_current)
        output_resistance = (sf_output_resistance * bias_transistor_resistance) / (sf_output_resistance + bias_transistor_resistance)

        total_capacitance = self.pixel.sf.source_cap + self.output_bus.cap_wire_per_m*self.output_bus_length + self.load_cap 
        + self.comparator_input_cap + self.SS_ADC.comparator.input_transistor.gate_cap
        total_resistance = output_resistance + self.output_bus.res_wire_per_m*self.output_bus_length

        t_slew = (max_voltage - min_voltage) * total_capacitance / self.bias_current
        t_settle = total_resistance * total_capacitance * 5 #typically the K_settle = 5 to get a stable settling
        readout_time = t_slew + t_settle
        return readout_time

    def compute_input_bus_delay_energy(self):
        #calculate the delay and the energy of the input bus
        wire_capacitance = self.input_bus.cap_wire_per_m*self.input_bus_length
        pixel_input_capacitance = self.num_cols * self.pixel.input_capacitance
        total_capacitance = wire_capacitance+pixel_input_capacitance
        total_resistance = self.input_bus.res_wire_per_m*self.input_bus_length
        delay = 2.3*total_capacitance*total_resistance/2
        energy = total_capacitance*(self.V_dd ** 2)

        return delay, energy
    
    def compute_rst_energy(self):
        #Epd+Efd
        total_capacitance = self.pixel.pd_node_cap + self.pixel.fd_cap
        energy = total_capacitance*self.pixel.rst_voltage*self.V_dd
        return energy

    #TODO:combine the following two functions into one function
    def compute_signal_readout_energy(self):
        #calculate the energy consumption of SF readout
        # total_capacitance = self.pixel.sf.source_cap + self.output_bus.cap_wire_per_m*self.output_bus_length + self.load_cap 
        # energy = 0.5*total_capacitance*(self.max_output_voltage**2)
        energy = self.bias_current*self.V_dd*self.readout_time
        return energy

    def compute_rst_readout_energy(self):
        # total_capacitance = self.pixel.sf.source_cap + self.output_bus.cap_wire_per_m*self.output_bus_length + self.load_cap 
        # energy = 0.5*total_capacitance*(self.max_output_voltage**2)
        energy = self.bias_current*self.V_dd*self.readout_time
        return energy
    
    # def compute_compare_energy(self, compare_time):
    #     #comparator's compare energy
    #     energy = self.V_dd*self.comparator.bias_current*compare_time
    #     return energy


    def compute_rc_lpf(self, reference_freq_hz, divider=10):
        fc = reference_freq_hz / divider
        rc = 1 / (2 * math.pi * fc)
        R = 1e3  # Ohms
        C = rc / R  # Farads
        return R,C
    
        
