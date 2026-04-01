from Phase_Frequency_Detector import PFD
from Charge_Pump import Charge_Pump
from LPF import LPF
from Voltage_Controlled_Oscillator import VCO
from Frequency_divider import Divider

# Phase-Locked Loop (PLL) class
# Models a complete PLL system with PFD, charge pump, LPF, VCO, and frequency divider
# 
# Inputs:
#   Voltage Parameters:
#     - bias_votlage (bias voltage in V)
#     - V_dd (supply voltage in V)
#     - V_ctrl (control voltage in V, for VCO)
#   Capacitance Parameters:
#     - CP_load_cap (charge pump load capacitance in F)
#     - LPF_load_cap (low-pass filter load capacitance in F)
#   Filter Parameters:
#     - LPF_resistance (low-pass filter resistance in Ω)
#   Timing Parameters:
#     - input_clk_freq (input clock frequency in Hz)
#     - output_clk_frequency (output clock frequency in Hz)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
# Outputs: PFD_dynamic_power (phase frequency detector power in W), CP_dynamic_power (charge pump power in W),
#          VCO_dynamic_power (voltage controlled oscillator power in W), FD_dynamic_power (frequency divider power in W),
#          total_dynamic_power (total PLL dynamic power in W)
class Phase_lock_loop:
    def __init__(self,CP_load_cap, LPF_resistance,LPF_load_cap, input_clk_freq, output_clk_frequency, bias_votlage, feature_size_nm, V_dd, V_ctrl):
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd
        self.input_clk_freq = input_clk_freq
        self.output_clk_frequency = output_clk_frequency
        #Bias voltage for the charge pump.
        self.bias_voltage = bias_votlage
        #Control voltage for the VCO, it is the voltage that is used to control the frequency of the VCO.
        self.V_ctrl = V_ctrl
        self.LPF_resistance = LPF_resistance
        self.CP_load_cap = CP_load_cap
        self.LPF_load_cap = LPF_load_cap
        #Activity factor is the factor that accounts for the activity of the CP circuit. If appply power gating.
        self.activity_factor = 1 #1 means the circuit is always active.

        #Initialize the PFD.
        self.PFD = PFD(
            clk_freq=self.input_clk_freq,
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        #Initialize the charge pump.
        self.CP = Charge_Pump(
            load_cap=self.CP_load_cap,
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd,
            bias_voltage= self.bias_voltage
        )

        #Initialize the low-pass filter.
        self.LPF = LPF(
            resistance=self.LPF_resistance,
            load_cap=self.LPF_load_cap
        )

        #Initialize the voltage controlled oscillator.
        self.VCO = VCO(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd,
            target_frequecny=self.output_clk_frequency,
            V_ctrl=self.V_ctrl
        )
        
        #Initialize the frequency divider.
        self.FD = Divider(
            input_clk_freq=self.output_clk_frequency,
            output_clk_frequency=self.input_clk_freq,
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        #Compute the dynamic power of the PFD.
        self.PFD_dynamic_power = self.compute_dynamic_energy(self.PFD.total_switch_cap + self.CP.total_input_cap)*self.input_clk_freq
        #Compute the dynamic power of the charge pump.
        self.CP_dynamic_power = self.CP.bias_current*self.V_dd*self.activity_factor
        #Compute the dynamic power of the voltage controlled oscillator.
        self.VCO_dynamic_power = self.compute_dynamic_energy(self.VCO.total_switch_cap + self.FD.input_cap) * self.output_clk_frequency
        #Compute the dynamic power of the frequency divider.
        self.FD_dynamic_power = 0
        for i in range(int(self.FD.num_of_stage)+1):
            self.FD_dynamic_power += self.compute_dynamic_energy(self.FD.total_switch_cap/self.FD.num_of_stage)*self.output_clk_frequency/(2.0**(i+1))

        self.total_dynamic_power = self.CP_dynamic_power+self.PFD_dynamic_power+self.VCO_dynamic_power+self.FD_dynamic_power

    def compute_dynamic_energy(self, total_cap):
        return total_cap*(self.V_dd**2)
    
# Define test parameters
# feature_size_nm = 65
# V_dd = 1.8
# input_clk_freq = 50e6          # 50 MHz reference clock
# output_clk_frequency = 4*input_clk_freq   # 200 MHz VCO output
# bias_voltage = 1.2
# V_ctrl = 0.6
# CP_load_cap = 5e-15            # 5 fF
# LPF_resistance = 10e3          # 10 kΩ
# LPF_load_cap = 10e-12          # 10 pF

# # Instantiate PLL
# pll = Phase_lock_loop(
#     CP_load_cap=CP_load_cap,
#     LPF_resistance=LPF_resistance,
#     LPF_load_cap=LPF_load_cap,
#     input_clk_freq=input_clk_freq,
#     output_clk_frequency=output_clk_frequency,
#     bias_votlage=bias_voltage,
#     feature_size_nm=feature_size_nm,
#     V_dd=V_dd,
#     V_ctrl=V_ctrl
# )

# # Print power consumption results
# print("PFD Dynamic Power: {:.2f} µW".format(pll.PFD_dynamic_power * 1e6))
# print("Charge Pump Dynamic Power: {:.2f} µW".format(pll.CP_dynamic_power * 1e6))
# print("VCO Dynamic Power: {:.2f} µW".format(pll.VCO_dynamic_power * 1e6))
# print("Frequency Divider Dynamic Power: {:.2f} µW".format(pll.FD_dynamic_power * 1e6))
# print("Total PLL Dynamic Power: {:.2f} µW".format(pll.total_dynamic_power * 1e6))