from parameter_class import NMOS, PMOS
import math
import numpy as np

# Ramp generator class
# Models a ramp generator circuit used in single-slope ADC for reference voltage generation
# 
# Inputs:
#   Voltage Parameters:
#     - bias_voltage (bias voltage in V)
#     - V_dd (supply voltage in V)
#   Capacitance Parameters:
#     - load_cap (load capacitance in F)
#   ADC Configuration:
#     - adc_resolution (ADC resolution in bits)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: bias_current (bias current in A), reset_time (reset time in s), total_cap (total capacitance in F),
#          discharge_energy (discharge energy in J), charge_energy (charge energy in J), CDS_time (CDS time in s),
#          ADC_time (ADC conversion time in s)
# It is a versy basic ramp generator circuit, is has a source transistor to provide bias current, a capacitor to store the charge, and generate a ramp voltage. and a reset transistor to reset the voltage of the laod.
class Ramp_generator:
    def __init__(self, bias_voltage, load_cap, feature_size_nm, V_dd, input_clk_freq, adc_resolution):
        
        self.V_dd = V_dd
        self.bias_voltage = bias_voltage
        self.load_cap = load_cap
        #Output voltage of the source follower.
        self.sf_out_voltage = 0 #Need to update this value at the top class

        self.source_transistor = NMOS (
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.rst_transistor = NMOS (
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        # self.switch_transistor = NMOS (
        #     tech_params=feature_size_nm,
        #     width=feature_size_nm*2,
        #     length=feature_size_nm,
        #     multiplier=1
        # )

        # Compute the reset voltage, it is the voltage for the reset transistor to reset the voltage of the load.
        self.rst_voltage = (self.V_dd-self.rst_transistor.v_th) * 0.9
        # Compute the bias current of the source transistor.
        self.bias_current = self.compute_current_source()
        # The CDS and ADC time are the time for the ramp voltage form full to null.
        # CDS time is the ramp time to perfrom digital CDS. ADC time is the ramp time to sample the ADC input signal.
        self.CDS_time = self.ADC_time = 1/input_clk_freq*(2**adc_resolution)
        # Compute the reset time, it is the time for the ramp voltage to reach to 90% of the Vgs-Vth of the reset transistor.
        self.reset_time = self.compute_reset_time(0)

        #switch and rst transistor has the same property, the total parasitic = 3*drain(or source)of the transistor. 2 from on transistor, and 1 fromt the off transistor
        self.total_cap = self.load_cap+self.rst_transistor.drain_cap*3 

        # Compute the discharge energy.
        self.discharge_energy = self.compute_energy(self.total_cap, self.rst_voltage,0)
        +self.compute_energy(self.total_cap, self.rst_voltage, self.sf_out_voltage)
        self.charge_energy = self.discharge_energy

    # Compute the bias current of the source transistor.
    def compute_current_source(self): #TODO:accurate VDS
        I = 0.5 * self.source_transistor.un * self.source_transistor.Cox * ((self.bias_voltage - self.source_transistor.v_th) **2)*(1+self.source_transistor.lambda_param*self.bias_voltage)
        return I

    # Compute the reset time, it is the time for the ramp voltage to reach to 90% of the Vgs-Vth of the reset transistor.
    def compute_reset_time(self, initial_voltage):
        V_initial = initial_voltage
        for val in np.arange(1e-12, 1.0001e-8, 1e-12):
            Vx = self.V_dd - V_initial #Vds = Vgs
            Vth = self.rst_transistor.v_th #updated Vth due to body effect
            
            #because the Vgs always equal to Vds, the reset transistor are in Saturation all the time
            I = 0.5 * self.rst_transistor.un * self.rst_transistor.Cox * ((Vx - Vth) **2)*(1+self.rst_transistor.lambda_param*Vx)
            V_initial = V_initial + I*1e-12/(self.load_cap + self.source_transistor.drain_cap + self.rst_transistor.drain_cap) #TODO: accurate cap information
            if V_initial >= self.rst_voltage: #when the rst voltage reach to 90% of the Vgs-Vth, stop reseting
                return val
        return val

    # Compute the energy 
    def compute_energy(self, Cap, Voltage1, Voltage2):
        energy = 0.5*Cap*(Voltage1**2 - Voltage2**2)
        return energy