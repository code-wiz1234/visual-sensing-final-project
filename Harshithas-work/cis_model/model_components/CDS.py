from Comparator import Comparator

# Correlated Double Sampling (CDS) class
# Models CDS circuit used to reduce noise in CIS readout by sampling reset and signal levels
# 
# Inputs:
#   Voltage Parameters:
#     - V_rst (reset voltage in V), V_sig (signal voltage in V)
#     - comparator_bias_voltage (comparator bias voltage in V)
#     - V_dd (supply voltage in V)
#   Capacitance Parameters:
#     - comaprator_FB_cap (comparator feedback capacitance in F)
#     - comparator_load_cap (comparator load capacitance in F)
#     - comparator_input_cap (comparator input capacitance in F)
#   Array Configuration:
#     - num_cols (number of columns), num_rows (number of rows)
#   Timing Parameters:
#     - frame_rate (frame rate in fps)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
#   Circuit Parameters:
#     - DC_gain (DC gain of comparator)
# Outputs: comparator_power (comparator power consumption in W), total_power (total CDS power in W),
#          CDS_delay (CDS operation delay in s), CDS_output_voltage (CDS output voltage in V)
class CDS:
    def __init__(self, V_rst, V_sig , comaprator_FB_cap ,
                  comparator_bias_voltage, comparator_load_cap, comparator_input_cap , 
                  num_cols , num_rows, frame_rate, feature_size_nm, V_dd, DC_gain):

        #intialize the comparator, the structure is like a feedback op-amp.
        self.comparator = Comparator(
            bias_voltage=comparator_bias_voltage,
            load_cap=comparator_load_cap,
            input_cap=comparator_input_cap,
            DC_gain = DC_gain,
            feature_size_nm=feature_size_nm,
            V_dd=V_dd
        )

        self.comparator_power = self.comparator.power*num_cols
        self.input_cap_energy = 0.5*comparator_input_cap*(V_rst**2 + (V_rst-V_sig)**2)
        self.settling_energy = 0.5*comparator_load_cap*((comaprator_FB_cap/comparator_input_cap)**2)*(V_sig-V_rst)**2
        self.total_power = (self.input_cap_energy+self.settling_energy)*frame_rate*num_cols*num_rows + self.comparator_power
        self.CDS_delay = self.comparator.AZ_time
        self.CDS_output_voltage = comaprator_FB_cap/comparator_input_cap * V_sig
        