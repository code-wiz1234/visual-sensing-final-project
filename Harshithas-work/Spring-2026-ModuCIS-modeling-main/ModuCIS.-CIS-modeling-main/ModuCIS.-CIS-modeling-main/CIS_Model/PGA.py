from Comparator import Comparator

# Programmable Gain Amplifier (PGA) class
# Models a PGA circuit used for signal amplification in CIS readout chains
# 
# Inputs:
#   Voltage Parameters:
#     - bias_voltage (bias voltage in V)
#     - V_dd (supply voltage in V)
#   Capacitance Parameters:
#     - input_cap (input capacitance in F)
#   Array Configuration:
#     - num_cols (number of columns), num_rows (number of rows)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#     - frame_rate (frame rate in fps)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
#   Circuit Parameters:
#     - DC_gain (DC gain)
# Outputs: input_clk_freq (input clock frequency in Hz),
#          latency (PGA latency in s), power (PGA power consumption in W)
# The structure of the PGA is like a feedback op-amp. In this model, we consider the static power. 
class PGA:
    def __init__(self, bias_voltage, input_clk_freq, input_cap, DC_gain, feature_size_nm, V_dd, frame_rate, num_cols, num_rows):
        self.input_cap = input_cap
        self.comparator = Comparator(
            bias_voltage = bias_voltage, 
            load_cap = 100e-15, 
            input_cap = input_cap, 
            DC_gain = DC_gain, 
            feature_size_nm = feature_size_nm, 
            V_dd = V_dd
        )

        self.input_clk_freq = input_clk_freq
        #2*DC_gain clks for amplification and sample, samp both sig and rst level so *2, PGA rst take
        # 2 clk so +2; *2 because two tap 
        #Compute the latency of the PGA, since it is a feedback op-amp, the latency is the same as the auto-zero time of the comparator.
        self.latency = self.comparator.AZ_time
        #Compute the total power of the PGAs
        self.power = self.comparator.power*num_cols