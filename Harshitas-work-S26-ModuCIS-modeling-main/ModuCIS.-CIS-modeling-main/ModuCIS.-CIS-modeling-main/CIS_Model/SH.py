import numpy as np
from parameter_class import NMOS, PMOS

# Sample and Hold (SH) circuit class
# Models a sample-and-hold circuit for sampling analog signals in CIS readout
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   Capacitance Parameters:
#     - load_cap (load capacitance in F)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: input_capacitance (input capacitance in F),
#          output_capacitance (output capacitance in F), energy (switching energy in J), delay (circuit delay in s)
# It is a basic SH circuit, is made up of a reset transistor, a select transistor (for read), a PMOS transistor (the switch), and a load cap to hold the signal
class SHCircuit:
    def __init__(self, V_dd, load_cap, feature_size_nm):
        self.V_dd = V_dd
        self.load_cap = load_cap #TODO: parasitic here?
        self.load_voltage = V_dd
        self.feature_size_nm = feature_size_nm

        self.rst_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.sel_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.PMOS = PMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        #Here I assume the switch is a TG circuit, it is composed of a Nmos an a Pmos, So a Pmos class is called here
        #Compute the input capacitance of the SH circuit. actually counting all the parasitic capacitance, and the load capacitance.
        self.input_capacitance = self.compute_input_cap() #used for readout time calculation
        self.output_capacitance = self.compute_output_cap()
        #Compute the dynamic energy for sampling 1 signal.
        self.energy = 1/2*self.input_capacitance*(self.load_voltage**2)*2
        self.delay = 0

    #TODO: double check
    def compute_input_cap(self):
        input_cap = self.rst_transistor.source_cap + self.PMOS.source_cap + self.load_cap + self.sel_transistor.drain_cap + self.PMOS.drain_cap
        return input_cap
    
    #TODO: double check
    def compute_output_cap(self): 
        output_cap = self.load_cap + self.sel_transistor.source_cap + self.PMOS.source_cap
        return output_cap

