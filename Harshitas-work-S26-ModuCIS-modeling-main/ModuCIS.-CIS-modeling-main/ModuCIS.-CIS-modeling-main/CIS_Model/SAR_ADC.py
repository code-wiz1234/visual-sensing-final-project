from Flip_Flop import Flip_Flop
from Comparator import Comparator

# Successive Approximation Register ADC (SAR_ADC) class
# Models a SAR ADC used in CIS readout circuits with comparator, DAC, and SAR logic components
# 
# Inputs:
#   Voltage Parameters:
#     - max_output_voltage (maximum output voltage in V)
#     - comparator_bias_voltage (comparator bias voltage in V)
#     - V_dd (supply voltage in V)
#   ADC Configuration:
#     - adc_resolution (ADC resolution in bits)
#   Capacitance Parameters:
#     - comparator_load_cap (comparator load capacitance in F)
#     - comparator_input_cap (comparator input capacitance in F)
#   Array Configuration:
#     - num_cols (number of columns), num_rows (number of rows)
#   Timing Parameters:
#     - frame_rate (frame rate in fps)
#     - input_clk_freq (input clock frequency in Hz)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
#   Circuit Parameters:
#     - DC_gain (DC gain of comparator)
# Outputs: comparator_power (comparator power in W), DAC_power (DAC power in W), SAR_power (SAR logic power in W),
#          total_power (total ADC power in W), delay (ADC conversion delay in s)
class SAR_ADC:
    def __init__(self, max_output_voltage, adc_resolution,
                  comparator_bias_voltage, comparator_load_cap, comparator_input_cap , 
                  num_cols, num_rows, frame_rate,input_clk_freq, feature_size_nm, V_dd, DC_gain):

        self.V_dd = V_dd
        #Supply voltage.
        #DC gain of the comparator.
        self.DC_gain = DC_gain
        self.frame_rate = frame_rate
        #Unit capacitance of the DAC.
        self.DAC_unit_cap = 10e-15

        #Initialize the comparator.
        self.comparator = Comparator(
            bias_voltage=comparator_bias_voltage,
            load_cap=comparator_load_cap,
            input_cap=comparator_input_cap,
            DC_gain = self.DC_gain,
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        #Initialize the DAC.
        self.DAC = DAC(
            unit_cap = self.DAC_unit_cap, 
            ADC_resolution = adc_resolution, 
            V_dd = V_dd, 
            V_ref = max_output_voltage
        )

        #Initialize the SAR logic.
        self.SAR = SAR(
            feature_size_nm=feature_size_nm,
            adc_resolution = adc_resolution,
            V_dd=V_dd, 
            input_clk_freq=input_clk_freq
        )

        #Compute the totalpower of the comparators.
        self.comparator_power = self.comparator.power*num_rows
        #Compute the power of the DAC.
        self.DAC_power = self.DAC.total_energy*self.frame_rate*num_cols*num_rows
        #Compute the power of the SAR logic.
        self.SAR_power = self.SAR.total_energy*num_cols*num_rows*frame_rate
        #Compute the total power of the all SAR ADCs.
        self.total_power = self.DAC_power+self.SAR_power + self.comparator_power
    

        #sense 1 bit need cone clk, extra 1 clk counts for sampling.
        self.delay = (adc_resolution + 1)/input_clk_freq


# Digital-to-Analog Converter (DAC) class
# Models a capacitor-based DAC used in SAR ADC for reference voltage generation
# 
# Inputs: unit_cap (unit capacitor value in F), ADC_resolution (ADC resolution in bits), V_dd (supply voltage in V),
#         V_ref (reference voltage in V)
# Outputs: total_cap (total capacitance in F),
#          total_energy (total switching energy in J)
# It is a basic DAC circuit, is made up of a capacitor array.
class DAC:
    def __init__(self, unit_cap, ADC_resolution, V_dd, V_ref):
        self.V_dd = V_dd
        self.unit_cap = unit_cap
        self.adc_resolution = ADC_resolution
        self.V_ref = V_ref
        
        self.total_cap = 2**ADC_resolution * self.unit_cap
        self.total_energy = 0.5*self.total_cap*self.V_ref**2


# Successive Approximation Register (SAR) class
# Models the SAR logic circuit that controls the conversion process in SAR ADC
# 
# Inputs: feature_size_nm (process technology node in nm), adc_resolution (ADC resolution in bits),
#         V_dd (supply voltage in V), input_clk_freq (input clock frequency in Hz)
# Outputs: num_of_stages (number of SAR stages),
#          input_cap (input capacitance in F), flip_flop_switch_cap (flip-flop switching capacitance in F),
#          total_energy (total switching energy in J)
# It is a basic SAR logic circuit, is made up of a flip-flop array.
class SAR:
    def __init__(self, feature_size_nm,  adc_resolution,  V_dd, input_clk_freq):
        self.V_dd = V_dd
        self.feature_size_nm = feature_size_nm
        self.input_clk_freq = input_clk_freq
        #Number of stages of the SAR logic.
        self.num_of_stages = 2*(adc_resolution + 1)
        #Activity factor of the SAR logic.
        self.activity_factor = 2/adc_resolution
        #Initialize the flip-flop.
        self.FF = Flip_Flop(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )
        #Compute the input capacitance of the SAR logic.
        self.input_cap = self.FF.D_input_cap
        #Compute the switching capacitance of the SAR logic.
        self.flip_flop_switch_cap = self.FF.total_switch_Cap
        #Compute the total energy of the SAR logic.
        self.total_energy = self.compute_dynamic_energy(self.flip_flop_switch_cap) * self.num_of_stages * adc_resolution
        # for i in self.num_of_stages:
        #     #for the SAR logic each stage has 2 FFs, so x2 at the end
        #     self.total_power += 1/(2*i) * self.compute_dynamic_energy(self.flip_flop_switch_cap)*self.input_clk_freq*2
        
    def compute_dynamic_energy(self, total_cap):
        return total_cap*(self.V_dd**2)