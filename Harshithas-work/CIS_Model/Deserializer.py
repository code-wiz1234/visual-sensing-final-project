from Flip_Flop import Flip_Flop

# Deserializer class
# Models a deserializer circuit for converting serial data to parallel format
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
#   Data Configuration:
#     - output_bits (number of output parallel bits)
# Outputs: top_power (top stage power in W),
#          bot_power (bottom stage power in W), total_power (total deserializer power in W)
# A basic deserializer that is made up of flip-flops. As a fully digital circuit without bias current, it only considers the dynamic power.
# The deserializer is used to convert the serial data to parallel data. Mainly used in Coded Exposure Designs.
class Deserializer:
    def __init__(self, input_clk_freq, feature_size_nm, V_dd, output_bits):
        self.V_dd = V_dd
        self.flip_flop = Flip_Flop(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        #Compute the top and bottom stage power. deserializer has two stages (up & down). Each stage has the same number of flip-flops, but with different working frequency.
        self.top_power = self.compute_dynamic_energy(self.flip_flop.total_switch_Cap) * input_clk_freq * output_bits
        self.bot_power = self.compute_dynamic_energy(self.flip_flop.total_switch_Cap) * input_clk_freq/output_bits * output_bits
        self.total_power = self.top_power + self.bot_power

    def compute_dynamic_energy(self, total_cap):
        return total_cap*(self.V_dd**2)