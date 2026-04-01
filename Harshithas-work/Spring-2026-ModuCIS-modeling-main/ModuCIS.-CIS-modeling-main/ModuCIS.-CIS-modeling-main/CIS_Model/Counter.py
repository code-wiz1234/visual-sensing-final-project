from Flip_Flop import Flip_Flop

# Counter class
# Models a binary counter circuit used in single-slope ADC for counting ramp cycles
# 
# Inputs:
#   Voltage Parameters:
#     - V_dd (supply voltage in V)
#   ADC Configuration:
#     - ADC_resolution (ADC resolution in bits)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: num_of_stages (number of counter stages),
#          input_cap (input capacitance in F), flip_flop_switch_cap (flip-flop switching capacitance in F),
#          total_power (total counter power in W), count_energy (counting energy per operation in J)
# A basic counter that is made up of flip-flops. As a fully digital circuit without bias current, it only considers the dynamic power.
class Countner:
    def __init__(self, ADC_resolution,input_clk_freq,feature_size_nm, V_dd):
        self.ADC_resolution = ADC_resolution
        self.feature_size_nm = feature_size_nm
        #self.num_of_count = num_of_count
        self.V_dd = V_dd
        self.num_of_stages = ADC_resolution
        self.input_clk_freq = input_clk_freq

        self.FF = Flip_Flop(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        #Compute the input capacitance of the counter.
        self.input_cap = self.FF.D_input_cap
        #Compute the flip-flop switching capacitance.
        self.flip_flop_switch_cap = self.FF.total_switch_Cap - self.FF.clk_input_cap
        #initialize the total power.
        self.total_power = 0
        #Compute the counting energy for one count operation.
        self.count_energy = 0.5*self.compute_dynamic_energy(self.flip_flop_switch_cap)
        #Compute the total power of the counter. (Each stage has different count frequency.)
        for i in range(self.num_of_stages):
            self.total_power += self.compute_dynamic_energy(self.flip_flop_switch_cap)*self.input_clk_freq/(2**i)
        
        #self.total_power += self.compute_dynamic_energy(self.FF.clk_input_cap)*self.input_clk_freq
    
    def compute_dynamic_energy(self, total_cap):
        #Compute the dynamic energy for one count operation.
        return total_cap*(self.V_dd**2)