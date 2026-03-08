from parameter_class import NMOS, PMOS
import math
from wire import Wire

# Analog buffer class
# Models an analog buffer circuit with input/output capacitance and resistance calculations
# 
# Inputs: wire_unit_cap (wire capacitance per unit length in F/m), wire_unit_res (wire resistance per unit length in Ω/m),
#         feature_size_nm (process technology node in nm), V_dd (supply voltage in V), bias_voltage (bias voltage in V)
# Outputs: bias_current (bias current in A),
#          input_cap (input capacitance in F), output_cap (output capacitance in F), internal_cap (internal capacitance in F),
#          output_resistance (output resistance in Ω)
# This class is used to model the analog buffer circuit. It is actually an operational amplifier
class Analog_Buffer:
    def __init__(self, wire_unit_cap, wire_unit_res, feature_size_nm,V_dd , bias_voltage):
        self.V_dd = V_dd
        self.feature_size_nm = 130
        self.bias_voltage = bias_voltage
        self.wire_cap = wire_unit_cap
        self.wire_res = wire_unit_res

        self.input_transistor1 = NMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*2,
            length=self.feature_size_nm,
            multiplier=1
        )

        self.input_transistor2 = NMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*2,
            length=self.feature_size_nm,
            multiplier=1
        )

        self.bias_transistor = NMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*4,
            length=self.feature_size_nm,
            multiplier=1
        )

        self.pull_up = PMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*4,
            length=self.feature_size_nm,
            multiplier=1
        )

        #Compute the bias current of the analog buffer.
        self.bias_current =  0.5 * self.bias_transistor.un * self.bias_transistor.Cox *(self.bias_transistor.width/self.bias_transistor.length)* ((self.bias_voltage - self.bias_transistor.v_th) **2)*(1+self.bias_transistor.lambda_param*self.bias_voltage)
        #Compute the input capacitance of the analog buffer.
        self.input_cap = self.input_transistor1.drain_cap
        #Compute the output capacitance of the analog buffer.
        self.output_cap = self.input_transistor2.drain_cap+self.pull_up.drain_cap + self.input_transistor2.drain_cap
        #Compute the internal capacitance of the analog buffer.
        self.internal_cap = self.input_transistor2.drain_cap+self.pull_up.drain_cap+self.pull_up.gate_cap*2
        #Compute the output resistance of the analog buffer.
        self.output_resistance = self.compute_output_resistance()

    #Compute the output resistance of the analog buffer.
    def compute_output_resistance(self):
        gm1 = math.sqrt(self.input_transistor2.un * self.input_transistor2.Cox * (self.bias_transistor.width/self.bias_transistor.length))*(1+self.bias_transistor.lambda_param*self.bias_voltage)
        gds1 = self.input_transistor2.lambda_param*self.bias_current/2
        ro1 = 1/(gm1+gds1)

        gm2 = math.sqrt(self.pull_up.un * self.pull_up.Cox * (self.bias_transistor.width/self.bias_transistor.length))*(1+self.bias_transistor.lambda_param*(self.V_dd-2*self.bias_voltage))
        gds2 = self.pull_up.lambda_param*self.bias_current/2
        ro2 = 1/(gm2+gds2)

        return 1/((1/ro1)+(1/ro2))
    
# Analog buffer bus class
# Models a bus with optimal repeater placement for analog signal transmission
# 
# Inputs: wire_unit_cap (wire capacitance per unit length in F/m), wire_unit_res (wire resistance per unit length in Ω/m),
#         feature_size_nm (process technology node in nm, optional, default=65), V_dd (supply voltage in V, optional, default=1.8),
#         bias_voltage (bias voltage in V, optional, default=0.6)
# Outputs: repeaterSize (optimal repeater size multiplier),
#          repeaterSpacing (optimal repeater spacing in m), unit_energy (energy per unit length in J/m)
# Most of the functions are based on the destiny SRAM simulator.
class Analog_Buffer_Bus:
    def __init__(self, wire_unit_cap, wire_unit_res, feature_size_nm = 65,V_dd = 1.8, bias_voltage = 0.6):
        self.V_dd = V_dd
        self.feature_size_nm = feature_size_nm
        self.bias_voltage = bias_voltage
        self.wire_cap = wire_unit_cap
        self.wire_res = wire_unit_res
        self.find_optimal_repeater()
        self.unit_energy = self.compute_unit_energy()

    #Find the optimal repeater size and spacing for the analog buffer.
    def find_optimal_repeater(self):
        self.repeaterSize = 1
        size = self.feature_size_nm*self.repeaterSize

        AB = Analog_Buffer(
            wire_unit_cap=self.wire_cap,
            wire_unit_res=self.wire_res,
            feature_size_nm=size,
            V_dd=self.V_dd,
            bias_voltage=self.bias_voltage
        )
       
        input_cap = AB.input_cap
        output_cap = AB.output_cap
        output_res = AB.output_resistance
        
        self.repeaterSize = math.sqrt(output_res * self.wire_cap / input_cap / self.wire_res)
        self.repeaterSpacing = math.sqrt(2 * output_res * (output_cap + input_cap) / (self.wire_res * self.wire_cap))
    
    #Compute the unit delay of the analog buffer.
    def compute_unit_delay(self):
        size = self.feature_size_nm*self.repeaterSize

        AB = Analog_Buffer(
            feature_size_nm=size,
            V_dd=self.V_dd,
            bias_voltage=self.bias_voltage
        )
       
        input_cap = AB.input_cap
        internal_cap = AB.internal_cap
        output_cap = AB.output_cap
        output_res = AB.output_resistance

        wire_cap = self.wire_cap * self.repeaterSpacing

        wire_cap = self.wire_cap * self.repeaterSpacing
        wire_res = self.wire_res * self.repeaterSpacing

        tau = output_res * (input_cap + output_cap + internal_cap) + output_res * wire_cap + wire_res * output_cap + 0.5 * wire_res * wire_cap
        return 0.693 * tau / self.repeaterSpacing
    
    #Compute the unit energy consumption of the analog buffer.
    def compute_unit_energy(self):
        size = self.feature_size_nm*self.repeaterSize

        AB = Analog_Buffer(
            feature_size_nm=size,
            V_dd=self.V_dd,
            bias_voltage=self.bias_voltage,
            wire_unit_cap=self.wire_cap,
            wire_unit_res=self.wire_res
        )
       
        input_cap = AB.input_cap
        internal_cap = AB.internal_cap
        output_cap = AB.output_cap
        wire_cap = self.wire_cap * self.repeaterSpacing

        switching_energy = (input_cap + output_cap + wire_cap + internal_cap) * self.V_dd ** 2
        return switching_energy / self.repeaterSpacing