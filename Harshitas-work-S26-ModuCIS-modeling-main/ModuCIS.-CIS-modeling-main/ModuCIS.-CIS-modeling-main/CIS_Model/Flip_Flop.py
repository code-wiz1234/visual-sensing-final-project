from parameter_class import NMOS, PMOS
from digital_gate import INV, NAND

# Flip-flop class
# Models a D-type flip-flop circuit built from inverters and NAND gates
# 
# Inputs: feature_size_nm (process technology node in nm), V_dd (supply voltage in V)
# Outputs: clk_input_cap (clock input capacitance in F),
#          D_input_cap (data input capacitance in F), Q_output_cap (Q output capacitance in F),
#          Q_bar_output_Cap (Q-bar output capacitance in F), total_switch_Cap (total switching capacitance in F)
# A basic flip-flop that is made up of inverters and NAND gates. 
class Flip_Flop:
    def __init__(self, feature_size_nm, V_dd):
        self.V_dd= V_dd
        self.feature_size_nm = feature_size_nm

        self.inverter = INV(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.input_nand_top = NAND(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.input_nand_bot = NAND(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.output_nand_top = NAND(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.output_nand_bot = NAND(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        #Compute the input capacitance of the flip-flop connected to the reference clock.
        self.clk_input_cap = self.input_nand_bot.input_cap_0 + self.input_nand_top.input_cap_0
        #Compute the data input capacitance of the flip-flop connected to the data input.
        self.D_input_cap = self.input_nand_top.input_cap_1 + self.inverter.input_cap
        #Compute the Q output capacitance of the flip-flop connected to the Q output.
        self.Q_output_cap = self.output_nand_top.output_cap + self.output_nand_bot.input_cap_0
        #Compute the Q-bar output capacitance of the flip-flop connected to the Q-bar output.
        self.Q_bar_output_Cap = self.output_nand_bot.output_cap + self.output_nand_top.input_cap_0
        #Compute the total switching capacitance of the flip-flop.
        self.total_switch_Cap = self.clk_input_cap + self.Q_bar_output_Cap + self.Q_output_cap
        + self.output_nand_top.input_cap_1 + self.output_nand_bot.input_cap_1 + self.input_nand_top.output_cap
        + self.input_nand_bot.output_cap