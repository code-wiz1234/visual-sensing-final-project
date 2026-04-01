from digital_gate import INV
from parameter_class import NMOS, PMOS

# Latch class
# Models an in-pixel latch circuit for storing pixel data
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   Timing Parameters:
#     - clk_freq (clock frequency in Hz)
#   Data Configuration:
#     - num_bit (number of bits)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: transistor_power (transistor switching power in W),
#          inv_power (inverter power in W), total_power (total latch power in W)
# A basic latch that is made up of inverters and NMOS/PMOS transistors. Used in coded exposure designs to store the "coded" information. It is structure is like SRAM cell.
class latch:
    def __init__(self, clk_freq, feature_size_nm, V_dd, num_bit):
        self.num_bit = num_bit
        self.inv = INV(
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        self.nmos = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.pmos = PMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )
        self.Vgs = V_dd
        #Compute the total dynamic power of the latch.  
        self.transistor_power = 1/2*self.pmos.gate_cap*(self.Vgs**2)*clk_freq * num_bit + 1/2*self.nmos.gate_cap*(self.Vgs**2)*clk_freq * num_bit
        #Compute the dynamic power of the inverter.
        self.inv_power = 1/2*(self.inv.output_cap + self.inv.input_cap + self.nmos.drain_cap)*(V_dd**2)*2 * clk_freq * num_bit
        #Compute the total power of the latch.
        self.total_power = self.transistor_power + self.inv_power