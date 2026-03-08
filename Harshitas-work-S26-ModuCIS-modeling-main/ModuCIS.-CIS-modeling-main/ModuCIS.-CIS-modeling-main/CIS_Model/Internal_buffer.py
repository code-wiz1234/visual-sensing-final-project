from parameter_class import NMOS, PMOS
from digital_gate import INV
import math

# Repeater (internal buffer) class
# Models a repeater circuit for signal buffering in long interconnects
# 
# Inputs: feature_size_nm (process technology node in nm), capWirePerUnit (wire capacitance per unit length in F/m),
#         resWirePerUnit (wire resistance per unit length in Ω/m), V_dd (supply voltage in V),
#         repeaterSize (repeater size multiplier, optional, default=1), repeaterSpacing (repeater spacing in m, optional, default=1)
# Outputs: repeaterSize (optimal repeater size multiplier),
#          repeaterSpacing (optimal repeater spacing in m), unit_delay (delay per unit length in s/m), unit_energy (energy per unit length in J/m)
# Repeaters are used to buffer the signal in the long interconnects. This code is based on the destiny SRAM simulator's work
class Repeater:
    def __init__(self, feature_size_nm, capWirePerUnit, resWirePerUnit,V_dd, repeaterSize=1, repeaterSpacing=1):
        self.feature_size_nm = feature_size_nm
        self.capWirePerUnit = capWirePerUnit
        self.resWirePerUnit = resWirePerUnit
        self.repeaterSize = repeaterSize
        self.repeaterSpacing = repeaterSpacing
        self.MIN_NMOS_SIZE = 1
        self.V_dd = V_dd
        self.find_optimal_repeater()
        self.unit_delay = self.compute_unit_delay()
        self.unit_energy = self.compute_unit_energy()

    def find_optimal_repeater(self):
        nmos_size = self.feature_size_nm*2*self.repeaterSize
        pmos_size = self.feature_size_nm*4*self.repeaterSize

        inv = INV(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        inv.pmos_width = pmos_size
        inv.nmos_width = nmos_size
       
        input_cap = inv.input_cap
        output_cap = inv.output_cap
        output_res = inv.resistance

        self.repeaterSize = math.sqrt(output_res * self.capWirePerUnit / input_cap / self.resWirePerUnit)
        self.repeaterSpacing = math.sqrt(2 * output_res * (output_cap + input_cap) / (self.resWirePerUnit * self.capWirePerUnit))

    def compute_unit_delay(self):
        nmos_size = self.feature_size_nm*2*self.repeaterSize
        pmos_size = self.feature_size_nm*4*self.repeaterSize

        inv = INV(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        inv.pmos_width = pmos_size
        inv.nmos_width = nmos_size
       
        input_cap = inv.input_cap
        output_cap = inv.output_cap
        output_res = inv.resistance

        wire_cap = self.capWirePerUnit * self.repeaterSpacing
        wire_res = self.resWirePerUnit * self.repeaterSpacing

        tau = output_res * (input_cap + output_cap) + output_res * wire_cap + wire_res * output_cap + 0.5 * wire_res * wire_cap
        return 0.693 * tau / self.repeaterSpacing

    def compute_unit_energy(self):
        nmos_size = self.feature_size_nm*2*self.repeaterSize
        pmos_size = self.feature_size_nm*4

        inv = INV(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        inv.pmos_width = pmos_size
        inv.nmos_width = nmos_size
       
        input_cap = inv.input_cap
        output_cap = inv.output_cap
        wire_cap = self.capWirePerUnit * self.repeaterSpacing

        switching_energy = (input_cap + output_cap + wire_cap) * self.V_dd ** 2
        return switching_energy / self.repeaterSpacing
