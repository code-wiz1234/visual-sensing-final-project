from Flip_Flop import Flip_Flop

# Input driver class
# Models an input driver circuit for driving control signals in CIS arrays
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   Array Configuration:
#     - num_rows (number of rows)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: ff_switch_cap (flip-flop switching capacitance in F),
#          total_power (total input driver power in W)
# A basic input driver that is made up of flip-flops. Used to pass to control signals in CIS arrays.
class Input_Driver:
    def __init__(self, input_clk_freq,num_rows, feature_size_nm,V_dd):
        self.V_dd = V_dd
        #Compute the number of flip-flops in the input driver. num_rows is the number of rows in the CIS array.
        self.num_ffs = num_rows
        self.feature_size_nm = feature_size_nm
        self.input_clk_freq = input_clk_freq

        self.flip_flop = Flip_Flop(
            V_dd=self.V_dd,
            feature_size_nm=self.feature_size_nm
        )

        #Compute the total switching capacitance of the input driver.
        self.ff_switch_cap = self.flip_flop.total_switch_Cap* self.num_ffs
        #Compute the total power of the input driver.
        self.total_power = self.compute_input_driver_power()
    
    def compute_input_driver_power(self):
        # print("cap input driver", self.ff_switch_cap*self.num_ffs)
        return 1/2*self.ff_switch_cap*(self.V_dd**2)*self.input_clk_freq

