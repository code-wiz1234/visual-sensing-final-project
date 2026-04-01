from Flip_Flop import Flip_Flop
from digital_gate import NAND, INV

# Row logic class
# Models row selection logic circuits for CIS array row addressing
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: and_gate_power (AND gate power in W), flip_flop_power (flip-flop power in W), nand_power (NAND gate power in W),
#          total_power (total row logic power in W)
class Row_logic:
    def __init__(self, input_clk_freq, feature_size_nm, V_dd):
        self.flip_flop = Flip_Flop(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )
        
        self.nand = NAND(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        self.inv = INV(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        self.and_gate_power = 1/2*(self.nand.total_switch_cap + self.inv.total_switch_cap)*(V_dd**2)*input_clk_freq
        self.flip_flop_power = 1/2*(self.flip_flop.total_switch_Cap)*(V_dd**2)*input_clk_freq
        self.nand_power = 1/2*(self.nand.total_switch_cap)*(V_dd**2)*input_clk_freq
        self.total_power = self.and_gate_power + self.flip_flop_power + self.nand_power