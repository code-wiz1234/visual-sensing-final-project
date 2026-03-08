from Photodiode import Photodiode
import math
#TODO:RESET Noise: only considered in 3T APS

# Shot noise class
# Models shot noise (quantum noise) in photodiodes due to statistical nature of photon detection
# 
# Inputs: area_um2 (photodiode area in μm²), reverse_bias (reverse bias voltage in V), lambda_m (wavelength in m),
#         eta (quantum efficiency), optical_power (optical power density in W/m²), exposure_time (exposure time in s),
#         feature_size_nm (process technology node in nm)
# Outputs: shot_noise_square (shot noise variance in electrons²), energy_per_photo (energy per photon in J),
#          power_per_pixel (optical power per pixel in W), temp (temporary calculation variable)
class Shot_Noise:
    def __init__(self, area_um2 ,reverse_bias,lambda_m ,eta,optical_power, exposure_time, feature_size_nm):
        #Constants for the photodiode.
        self.EPSILON_0 = 8.854e-12  # F/m
        self.EPSILON_SI = 11.7 * self.EPSILON_0  # Permittivity of silicon (F/m)
        self.Q = 1.602e-19  # Electron charge (C)
        self.H = 6.626e-34        # Planck's constant (J·s)
        self.C = 3e8 
        self.feature_size_nm =feature_size_nm
        self.area_um2 = area_um2
        self.area_m2 = area_um2 * 1e-12
        self.reverse_bias = reverse_bias #it should be the Vdd - Vth_rst
        self.v_bi = 0.7  # Default built-in voltage
        self.eta = eta #conversion efficiency
        self.optical_power = optical_power #input power density per unit area
        self.exposure_time = exposure_time
        self.power_per_pixel = self.optical_power*self.area_m2
        self.lambda_m = lambda_m
        self.energy_per_photo = self.compute_energy_per_photo()
        self.temp = self.power_per_pixel*self.eta*self.exposure_time/self.energy_per_photo
        
        self.shot_noise_square = self.compute_photoelectron_count()

    #Compute the energy per photon.
    def compute_energy_per_photo(self):
        return self.H*self.C/self.lambda_m
    
    #Compute the number of photoelectrons.
    def compute_photoelectron_count(self):
        N_ph = self.power_per_pixel*self.eta*self.exposure_time/self.energy_per_photo
        return N_ph
    #I_pd = self.eta * self.E * self.area_m2 * (self.Q * self.lambda_m) / (self.H * self.C)*exposure_time

# Dark current noise class
# Models dark current noise in photodiodes due to thermally generated carriers
# 
# Inputs: area_um2 (photodiode area in μm²), exposure_time (exposure time in s),
#         dark_currnet_density (dark current density in A/cm², optional, default=14.75e-9)
# Outputs: dark_current (dark current in A), dark_current_noise_square (dark current noise variance in electrons²),
#          area_cm2 (photodiode area in cm²)
class Dark_Current_Noise:
    def __init__(self, area_um2,exposure_time, dark_currnet_density = 14.75e-9):
        self.Q = 1.602e-19 
        self.area_um2 = area_um2
        self.area_cm2 = area_um2 * 1e-6
        self.exposure_time = exposure_time
        self.dark_current_density = dark_currnet_density
        self.dark_current = self.dark_current_density*self.area_cm2
        self.dark_current_noise_square = self.compute_dark_current_noise()

    def compute_dark_current_noise(self):
        N_dark = self.dark_current*self.exposure_time/self.Q
        return N_dark


# Thermal noise class
# Models thermal (Johnson-Nyquist) noise in transistors and circuits
# 
# Inputs: transconductance (transistor transconductance in S), cap (capacitance in F),
#         temperature (temperature in K, optional, default=300)
# Outputs: V_thermal_noise (thermal noise voltage in V), thermal_noise_square (thermal noise variance in electrons²),
#          CG (conversion gain in V/electron)
class Thermal_Noise:
    def __init__(self, transconductance, cap, temperature = 300):
        self.K = 1.380649e-23 #Boltzmann Constant
        self.Q = 1.602e-19
        self.transconductance  =transconductance
        self.cap = cap
        self.temperature = temperature
        #Compute the conversion gain.
        self.CG = self.Q/cap
        #Compute the thermal noise in volts.
        self.V_thermal_noise = math.sqrt(self.compute_thermal_noise_in_volt())
        #Compute the thermal noise in electrons.
        self.thermal_noise_square = self.compute_thermal_noise()

    #Compute the thermal noise in volts.
    def compute_thermal_noise_in_volt(self):
        V_thermal = (4*self.temperature*self.K*2/3)/self.transconductance
        return V_thermal
    
    #Compute the thermal noise in electrons.
    def compute_thermal_noise(self):
        return (self.V_thermal_noise/self.CG)**2

# Random Telegraph Signal (RTS) noise class
# Models RTS noise caused by carrier trapping and detrapping in transistors
# 
# Inputs: trapping_time (carrier trapping time in s), detrapping_time (carrier detrapping time in s),
#         amplitude (RTS noise amplitude in electrons)
# Outputs: RTS_noise (RTS noise value in electrons)
class RTS_Noise:
    def __init__(self, trapping_time, detrapping_time, amplitude):
        self.trapping_time = trapping_time
        self.detrapping_time = detrapping_time
        self.amplitude = amplitude
        self.RTS_noise = self.compute_RTS_Noise()

    def compute_RTS_Noise(self):
        N_RTS = self.amplitude*math.sqrt((self.trapping_time*self.detrapping_time)/((self.trapping_time+self.detrapping_time)**2))
        return N_RTS

#FPN noise is cancled by the CDS, only consider when the CDS is not used
# class Fixed_Pattern_Noise:
#     def __init__(self, input_clk_freq,num_rows, feature_size_nm = 65,V_dd = 1.8):
#         feature_size_nm
#Like the FPN noise, the flicker noise is also cancled by the CDS, only consider when the CDS is not used
# class Flicker_Noise:
#     def __init__(self, input_clk_freq,num_rows, feature_size_nm = 65,V_dd = 1.8):
#         feature_size_nm

# Transfer noise class
# Models charge transfer noise in pixel readout circuits
# 
# Inputs: num_photoelectrons (number of photoelectrons), CTI (charge transfer inefficiency, optional, default=0.001)
# Outputs: transfer_noise_square (transfer noise variance in electrons²)
class Transfer_Noise:
    def __init__(self, num_photoelectrons,CTI=0.001):
        self.transfer_noise_square = num_photoelectrons*CTI

# Current shot noise class
# Models shot noise in current-carrying circuits
# 
# Inputs: current (bias current in A), transconductance (transconductance in S), input_cap (input capacitance in F),
#         load_cap (load capacitance in F), output_res (output resistance in Ω)
# Outputs: bandwidth (circuit bandwidth in Hz), noise_in_current_square (current noise variance in A²),
#          current_shot_noise_square (current shot noise variance in electrons²), CG (conversion gain in V/electron)
class Current_Shot_Noise:
    def __init__(self,current, transconductance, input_cap, load_cap, output_res):
        self.Q = 1.602e-19
        self.current = current
        self.transconductance = transconductance
        self.input_cap = input_cap
        #Compute the conversion gain.
        self.CG = self.Q/input_cap
        self.load_cap = load_cap
        self.output_res = output_res
        #Compute the bandwidth.
        self.bandwidth = self.compute_bandwidth()
        #Compute the noise in current square.
        self.noise_in_current_square = 2*self.Q*self.current*self.bandwidth
        #Compute the current shot noise square.
        self.current_shot_noise_square = self.compute_current_shot_noise()

    #Compute the bandwidth.
    def compute_bandwidth(self):
        pi = 3.1415926535
        return 1/(2*pi*self.output_res*self.load_cap)
    
    #Compute the current shot noise square.
    def compute_current_shot_noise(self):
        noise_in_current_square = 2*self.Q*self.current*self.bandwidth
        noise_in_electrons_square = noise_in_current_square/((self.transconductance*self.CG)**2)
        return noise_in_electrons_square

# class Read_Noise:
    
# ADC quantization noise class
# Models quantization noise introduced by analog-to-digital conversion
# 
# Inputs: ADC_resolution (ADC resolution in bits), voltage_range (ADC input voltage range in V), cap (capacitance in F)
# Outputs: RMS_quant_noise (RMS quantization noise in V), quant_noise (quantization noise in electrons),
#          quant_noise_square (quantization noise variance in electrons²), quant_step (quantization step in V),
#          CG (conversion gain in V/electron)
class ADC_Quantizing_Noise:
    def __init__(self, ADC_resolution, voltage_range, cap):
        self.Q = 1.602e-19
        self.ADC_resolution = ADC_resolution
        self.voltage_range = voltage_range
        #Compute the conversion gain.
        self.CG = self.Q/cap
        #Compute the Root mean square quantization noise.
        self.RMS_quant_noise = self.compute_RMS_quant_Noise()
        #Compute the quantization noise.
        self.quant_noise = self.RMS_quant_noise*cap/self.Q
        #Compute the quantization noise variance.
        self.quant_noise_square = (self.quant_noise ** 2)

    #Compute the Root mean square quantization noise.
    def compute_RMS_quant_Noise(self):
        #Compute the quantization step.
        self.quant_step = self.voltage_range/(2**self.ADC_resolution)
        return self.quant_step/(math.sqrt(12))


# System noise class
# Models overall system noise (placeholder for future implementation)
# 
# Inputs: input_clk_freq (input clock frequency in Hz), num_rows (number of rows),
#         feature_size_nm (process technology node in nm, optional, default=65), V_dd (supply voltage in V, optional, default=1.8)
# Outputs: (placeholder - not yet implemented)
class System_Noise:
    def __init__(self, input_clk_freq,num_rows, feature_size_nm = 65,V_dd = 1.8):
        feature_size_nm

# Reset noise class
# Models reset (kTC) noise in pixel reset operations
# 
# Inputs: cap (capacitance in F), temperature (temperature in K, optional, default=300)
# Outputs: reset_noise (reset noise in electrons)
class Reset_Noise:
    def __init__(self, cap,temperature=300):
        self.K = 1.380649e-23 #Boltzmann Constant
        #Compute the reset noise.
        self.reset_noise = math.sqrt(self.K*temperature/cap)