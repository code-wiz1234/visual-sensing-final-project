from parameter_class import NMOS, PMOS
import math
from Noise import Thermal_Noise

# Comparator class
# Models a comparator circuit with auto-zeroing capability for signal comparison in CIS readout
# 
# Inputs:
#   Voltage Parameters:
#     - bias_voltage (bias voltage in V)
#     - V_dd (supply voltage in V)
#   Capacitance Parameters:
#     - load_cap (load capacitance in F)
#     - input_cap (input capacitance in F)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
#   Circuit Parameters:
#     - DC_gain (DC gain)
# Outputs: bias_current (bias current in A), offset_voltage (offset voltage in V), AZ_time (auto-zero time in s),
#          output_cap (output capacitance in F), output_res (output resistance in Ω), Gm (transconductance in S),
#          power (comparator power in W), energy (switching energy in J)
# The structure of the comparator is like a feedback op-amp. In this model, we consider the static power. 
class Comparator:
    def __init__(self, bias_voltage, load_cap, input_cap, DC_gain, feature_size_nm, V_dd):
        
        self.V_dd = V_dd
        self.bias_voltage = bias_voltage
        self.load_cap = load_cap
        self.input_cap = input_cap
        self.DC_gain = DC_gain

        self.bias_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.input_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.pull_up_transistor = PMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*4,
            length=feature_size_nm,
            multiplier=1
        )

        self.switch_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )


        self.bias_current = self.compute_bias_current()
        #The Auto Zero voltage is due to process, so cannot have a accurate value.
        #just assume the max off-set voltage = V_dd
        self.offset_voltage = self.V_dd - self.switch_transistor.v_th
        #Compute the auto-zero time.
        self.AZ_time = self.compute_Auto_zero_time(self.V_dd - self.offset_voltage, 0)
        #Compute the output capacitance. Include all the parasitic capacitance at the output node.
        self.output_cap = self.load_cap + self.pull_up_transistor.drain_cap + self.input_transistor.drain_cap
        #Compute the bias current of the input and pull up transistors. one half of the total bias current.
        self.input_transistor.bias_current=0.5*self.bias_current
        #Compute the bias current of the pull up transistor. one half of the total bias current.    
        self.pull_up_transistor.bias_current=0.5*self.bias_current
        #Compute the output resistance of the comparator.
        self.output_res = self.compute_output_resistance()
        #Compute the transconductance of the input transistor.
        self.Gm = self.input_transistor.gm
        self.power = self.bias_current*V_dd
        #Compute the dynamic energy of the comparator output load..
        self.energy = 1/2*load_cap*(V_dd**2)

    def compute_output_resistance(self):
        output_res = 1/(1/(1/self.input_transistor.gm) + 1/(1/self.pull_up_transistor.gm))
        return output_res

    def compute_bias_current(self):
        bias_current = 0.5 * self.bias_transistor.un * self.bias_transistor.Cox * ((self.bias_voltage - self.bias_transistor.v_th) **2)*(1+self.bias_transistor.lambda_param*self.bias_voltage)
        return bias_current

    #Compute the auto-zero time of the comparator. The Auto zero is used for the SS ADC to remove the noise.
    def compute_Auto_zero_time(self, max_voltage, min_voltage):
        #compute the readout time
        #max_voltage: max possible voltage of the AZ cap
        #min_voltage: min possible voltage of the AZ cap, usually 0
        total_capacitance = self.input_cap + self.load_cap + self.input_transistor.gate_cap
        + self.input_transistor.drain_cap + self.pull_up_transistor.drain_cap + self.switch_transistor.drain_cap + self.switch_transistor.source_cap

        input_transistor_one_over_gm = 1/math.sqrt(2 * self.input_transistor.un * self.input_transistor.Cox * (self.input_transistor.width / self.input_transistor.length) * self.bias_current * 0.5) #1/gm, assume the current is half of the bias current
        input_transistor_resistance = 1/(self.input_transistor.lambda_param*self.bias_current*0.5)
        input_transistor_resistance = (input_transistor_one_over_gm * input_transistor_resistance)/(input_transistor_resistance+input_transistor_one_over_gm) #1/gm // rds

        output_resistance =  1/(self.pull_up_transistor.lambda_param*self.bias_current*0.5)
        total_resistance = output_resistance * input_transistor_resistance/(output_resistance + input_transistor_resistance) #TODO: consider the body effect of the diode transistor?
        t_slew = (max_voltage - min_voltage) * total_capacitance / self.bias_current
        t_settle = total_resistance * total_capacitance * 5 / (1 + self.DC_gain) #typically the K_settle = 5 to get a stable settling
        readout_time = t_slew + t_settle
        return readout_time
    
    def compute_Auto_zero_energy(self):
        energy = (self.input_cap+self.input_transistor.gate_cap+self.switch_transistor.drain_cap) * (self.offset_voltage ** 2)
        return energy