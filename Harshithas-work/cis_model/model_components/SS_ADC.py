from Counter import Countner
from Comparator import Comparator
from Ramp_generator import Ramp_generator

# Single-Slope ADC (SS_ADC) class
# Models a single-slope analog-to-digital converter used in CIS readout circuits
# Includes comparator, ramp generator, and counter components
# 
# Inputs:
#   Voltage Parameters:
#     - ramp_bias_voltage (ramp generator bias voltage in V)
#     - max_output_voltage (maximum output voltage in V)
#     - comparator_bias_voltage (comparator bias voltage in V)
#     - V_dd (supply voltage in V)
#   ADC Configuration:
#     - adc_resolution (ADC resolution in bits)
#   Capacitance Parameters:
#     - ramp_load_cap (ramp generator load capacitance in F)
#     - comparator_load_cap (comparator load capacitance in F)
#     - comparator_input_cap (comparator input capacitance in F)
#   Array Configuration:
#     - num_cols (number of columns), num_rows (number of rows)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#     - frame_rate (frame rate in fps)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: compare_power (comparator comparison power in W),
#          comparator_power (total comparator power in W),
#          ramp_generator_power (ramp generator power in W), counter_total_power (counter power in W),
#          adc_power (total ADC power in W)
class SS_ADC:
    def __init__(self, input_clk_freq, ramp_bias_voltage, ramp_load_cap, max_output_voltage, adc_resolution,
                  comparator_bias_voltage, comparator_load_cap, comparator_input_cap, 
                  num_cols, num_rows, frame_rate, feature_size_nm, V_dd):

        self.V_dd = V_dd
        #DC gain of the comparator. Assume very large
        self.DC_gain = 200
        #Number of columns of the CIS array.
        self.num_cols = num_cols
        #Number of rows of the CIS array.
        self.num_rows = num_rows
        #Frame rate of the CIS array.
        self.frame_rate = frame_rate
        #Initialize the comparator.
        self.comparator = Comparator(
            bias_voltage=comparator_bias_voltage,
            load_cap=comparator_load_cap,
            input_cap=comparator_input_cap,
            DC_gain = self.DC_gain,
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        #Clock frequency of the counter.
        self.counter_clk_freq  = input_clk_freq
        #Initialize the ramp generator.
        self.ramp_generator = Ramp_generator(
            bias_voltage=ramp_bias_voltage,
            load_cap=ramp_load_cap,
            feature_size_nm=feature_size_nm,
            V_dd=V_dd,
            input_clk_freq=input_clk_freq,
            adc_resolution=adc_resolution
        )

        # Update the source follower output voltage to the ramp generator class
        self.ramp_generator.sf_out_voltage = max_output_voltage

        #Initialize the counter.
        self.counter = Countner(
            ADC_resolution = adc_resolution,
            input_clk_freq= self.counter_clk_freq,
            feature_size_nm = feature_size_nm, 
            V_dd=V_dd
        )
        #Compute the energy of the ramp generator.  
        self.ramp_generator_energy = self.ramp_generator.discharge_energy+self.ramp_generator.charge_energy
        #Compute the total power of the counter.
        self.counter_total_power = self.counter.total_power*self.num_cols*self.num_rows*(self.ramp_generator.ADC_time+self.ramp_generator.CDS_time)*self.frame_rate
        #Compute the power of the comparator.
        #Compute the dynamic power of the comparator.
        self.comparator_power = self.comparator.compute_Auto_zero_energy()*self.frame_rate*self.num_cols*self.num_rows
        #Add the static power of the comaprator to the total comparator power.
        self.comparator_power += self.comparator.power*num_cols
        #Compute the power of the ramp generator.
        self.ramp_generator_power = self.ramp_generator_energy*self.frame_rate*self.num_rows
        #Compute the total power of the ADC.
        self.adc_power = self.comparator_power+self.ramp_generator_power+self.counter_total_power
