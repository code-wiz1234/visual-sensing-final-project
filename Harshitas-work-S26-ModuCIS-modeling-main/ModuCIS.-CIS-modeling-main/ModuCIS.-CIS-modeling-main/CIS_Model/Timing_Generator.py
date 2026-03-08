from Flip_Flop import Flip_Flop
from digital_gate import NAND, INV, NOR
from Counter import Countner
import math

# Pulse generator class
# Generates timing pulses with specified width using counter and logic gates
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#     - pulse_width (pulse width in clock cycles, optional, default=15)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: counter_power (counter power in W),
#          nand_power (NAND gate power in W), or_power (NOR/INV power in W), total_power (total power in W)
class pulse_generator:
    def __init__(self, input_clk_freq, feature_size_nm, V_dd, pulse_width = 15):
        
        self.counter = Countner(
            ADC_resolution = math.ceil((pulse_width + 1)**0.5),
            input_clk_freq = input_clk_freq,
            feature_size_nm = feature_size_nm,
            V_dd=V_dd
        )

        #2NAND
        self.nand = NAND(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        #1 NOR
        self.nor = NOR(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        #1 INV
        self.inv = INV(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )
        
        self.counter_power = self.counter.total_power
        self.nand_power = 1/2*(self.nand.total_switch_cap)*(V_dd**2)*input_clk_freq*2
        self.or_power = 1/2*(self.nor.total_switch_cap)*(V_dd**2)*input_clk_freq + 1/2*(self.inv.total_switch_cap)*(V_dd**2)*input_clk_freq
        self.total_power = self.counter_power + self.nand_power + self.or_power

# Time generator class
# Generates timing signals for CIS operations using pulse generator and flip-flops
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#     - denominator (frequency division denominator)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: pulse_generator_power (pulse generator power in W),
#          flip_flop_power (flip-flop power in W), or_power (NOR/INV power in W), inv_power (inverter power in W),
#          and1_power (first AND gate power in W), and2_power (second AND gate power in W), total_power (total power in W)
class time_generator:
    def __init__(self, input_clk_freq, feature_size_nm, V_dd, denominator):

        self.pulse_generator = pulse_generator(
            input_clk_freq = input_clk_freq/denominator,
            feature_size_nm = feature_size_nm,
            V_dd=V_dd, 
            pulse_width = denominator-1
        )

        #18 = denominator + 2
        self.flip_flop = Flip_Flop(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        #2NAND
        self.nand = NAND(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        #1 NOR
        self.nor = NOR(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        #4 INV
        self.inv = INV(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        self.pulse_generator_power = self.pulse_generator.total_power
        self.flip_flop_power = 1/2*(self.flip_flop.total_switch_Cap)*(V_dd**2)*(2+denominator)*input_clk_freq
        self.or_power = 1/2*(self.nor.total_switch_cap)*(V_dd**2)*input_clk_freq + 1/2*(self.inv.total_switch_cap)*(V_dd**2)*input_clk_freq
        self.inv_power = 1/2*(self.inv.total_switch_cap)*(V_dd**2)*input_clk_freq
        #one input with switch frequency = input clk freq, one = input clk freq/16(denominator)
        self.and1_power = 1/2*(self.nand.total_switch_cap - self.nand.input_cap_0)*(V_dd**2)*input_clk_freq + 1/2*(self.inv.total_switch_cap)*(V_dd**2)*input_clk_freq + 1/2*(self.nand.input_cap_0)*(V_dd**2)*input_clk_freq/denominator
        self.and2_power = 1/2*(self.nand.total_switch_cap)*(V_dd**2)*input_clk_freq + 1/2*(self.inv.total_switch_cap)*(V_dd**2)*input_clk_freq
        self.total_power = self.pulse_generator_power + self.flip_flop_power +self.or_power + self.inv_power + self.and1_power + self.and2_power