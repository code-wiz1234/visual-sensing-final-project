from parameter_class import NMOS, PMOS

# In-pixel control circuit class
# Models control logic circuits within pixel arrays
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   Timing Parameters:
#     - clk_freq (clock frequency in Hz)
#   Data Configuration:
#     - num_bit (number of control bits)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: total_power (total control circuit power in W),
#          num_ctrl (number of control signals)
# The input pixel control logic mainly used in Coded Exposure Designs. Refer to the paper form University of Toronto.
class ctrl:
    def __init__(self, clk_freq, feature_size_nm, V_dd, num_bit):
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
        #Compute the total dynamci power of the control circuit.
        self.total_power = (1/2*self.pmos.gate_cap*(self.Vgs**2)*clk_freq * 2 + 1/2*self.nmos.gate_cap*(self.Vgs**2)*clk_freq * 4 )* num_bit
        self.num_ctrl = num_bit