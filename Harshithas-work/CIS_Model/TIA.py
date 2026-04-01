from Comparator import Comparator

# Transimpedance Amplifier (TIA) class
# Models a TIA circuit used to convert photocurrent to voltage in CIS pixels
# 
# Inputs:
#   Voltage Parameters:
#     - bias_voltage (bias voltage in V)
#     - V_dd (supply voltage in V)
#   Capacitance Parameters:
#     - input_cap (input capacitance in F)
#   Array Configuration:
#     - num_cols (number of columns)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
#   Circuit Parameters:
#     - DC_gain (DC gain)
# Outputs: input_clk_freq (input clock frequency in Hz),
#          power (TIA power consumption in W)
# It is a basic TIA circuit, made of op-amp, has the same structure as the comparator. Used for the CNN PMW photoreceptors.
class TIA:
    def __init__(self, bias_voltage, input_clk_freq, input_cap, DC_gain, feature_size_nm, V_dd, num_cols):
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
        self.power = self.comparator.power*num_cols