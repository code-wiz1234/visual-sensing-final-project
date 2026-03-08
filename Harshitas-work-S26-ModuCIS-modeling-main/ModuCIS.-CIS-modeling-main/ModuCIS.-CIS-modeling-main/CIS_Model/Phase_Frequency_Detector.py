from parameter_class import NMOS, PMOS
from digital_gate import NAND, INV
from Flip_Flop import Flip_Flop

# Phase Frequency Detector (PFD) class
# Models a PFD circuit used in PLL to detect phase and frequency differences
# 
# Inputs: clk_freq (clock frequency in Hz), feature_size_nm (process technology node in nm), V_dd (supply voltage in V)
# Outputs: clk_input_cap_bot (bottom clock input capacitance in F),
#          clk_input_cap_top (top clock input capacitance in F), output_cap_top (top output capacitance in F),
#          output_cap_bot (bottom output capacitance in F), total_switch_cap (total switching capacitance in F)
# The typical structure of the PFD is two flip-flops and a NAND gate.
class PFD:
    def __init__(self, clk_freq, feature_size_nm, V_dd):
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd
        self.clk_freq = clk_freq #the reference clk frequ from the off chip

        self.FF_up = Flip_Flop(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.FF_down = Flip_Flop(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.NAND = NAND(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        #Compute the input and output capacitance of the PFD.
        self.clk_input_cap_bot = self.FF_down.clk_input_cap
        #Compute the input and output capacitance of the PFD.
        self.clk_input_cap_top = self.FF_up.clk_input_cap
        #Compute the output capacitance of the PFD.
        self.output_cap_top = self.FF_up.Q_output_cap + self.NAND.input_cap_0
        #Compute the output capacitance of the PFD.
        self.output_cap_bot = self.FF_up.Q_output_cap + self.NAND.input_cap_1
        #Compute the total switching capacitance of the PFD. Which can be used to compute the dynamic power of the PFD.
        self.total_switch_cap = self.FF_up.total_switch_Cap + self.FF_down.total_switch_Cap + self.NAND.input_cap_0
        + self.NAND.input_cap_1
