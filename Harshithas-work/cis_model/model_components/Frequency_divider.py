import math
from Flip_Flop import Flip_Flop
from digital_gate import INV
#TODO

# Frequency divider class
# Models a frequency divider circuit using flip-flops to divide clock frequency
# 
# Inputs: input_clk_freq (input clock frequency in Hz), output_clk_frequency (output clock frequency in Hz),
#         feature_size_nm (process technology node in nm), V_dd (supply voltage in V)
# Outputs: num_of_stage (number of divider stages),
#          input_cap (input capacitance in F), output_cap (output capacitance in F), total_switch_cap (total switching capacitance in F)
# A basic frequency divider that is made up of flip-flops. It is one of the core components in PLL.
class Divider:
    def __init__(self, input_clk_freq, output_clk_frequency, feature_size_nm, V_dd):
        self.V_dd = V_dd
        self.feature_size_nm = feature_size_nm
        self.input_clk_freq = input_clk_freq
        self.output_clk_frequency = output_clk_frequency
        self.temp = int(input_clk_freq/output_clk_frequency)

        #Compute the number of divider stages, based on the input and output clock frequency.
        if self.is_power_of_two(self.temp):
            self.num_of_stage = math.sqrt(input_clk_freq/output_clk_frequency)
        else:
            self.num_of_stage = int(math.sqrt(input_clk_freq/output_clk_frequency))
        # print("Number of divider stages: {:.2f}".format(self.num_of_stage))
        
        self.flip_flop = Flip_Flop(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        #Compute the input and output capacitance of the divider.   
        self.input_cap = self.flip_flop.clk_input_cap
        #Compute the output capacitance of the divider.
        self.output_cap = self.flip_flop.Q_output_cap
        #Compute the total switching capacitance of the divider.
        self.total_switch_cap = (self.flip_flop.total_switch_Cap+self.flip_flop.D_input_cap)*self.num_of_stage

    def is_power_of_two(self,n):
        return n > 0 and (n & (n - 1)) == 0