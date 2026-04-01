from digital_gate import INV

# Voltage Controlled Oscillator (VCO) class
# Models a VCO circuit using inverter chain for clock generation in PLL
# 
# Inputs: feature_size_nm (process technology node in nm), V_dd (supply voltage in V),
#         target_frequecny (target oscillation frequency in Hz), V_ctrl (control voltage in V)
# Outputs: num_inverter (number of inverter stages),
#          total_switch_cap (total switching capacitance in F), t_inv (inverter delay in s)
class VCO:
    def __init__(self, feature_size_nm, V_dd, target_frequecny, V_ctrl):
        #TODO: consider the feedback Opamp input votlage
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd #VDD is actually the controlled signal from the charge pump
        self.target_frequency = target_frequecny

        #I assueme the controlled voltage is around 1/3 V_dd
        if V_ctrl:
            self.V_ctrl = V_ctrl
        else:
            self.V_ctrl = self.V_dd/3

        self.inverter_chain = INV(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_ctrl
        )

        self.t_inv = self.inverter_chain.delay*V_ctrl
        self.num_inverter = self.compute_num_of_stage()
        self.total_switch_cap = self.num_inverter*(self.inverter_chain.input_cap+self.inverter_chain.output_cap)

    def compute_num_of_stage(self):
        return int(1/(2*self.target_frequency*self.t_inv))
    
    def compute_power(self):
        C_total = self.num_inverter*(self.inverter_chain.input_cap+self.inverter_chain.output_cap)
        return C_total*(self.V_ctrl**2)*self.target_frequency

# print("Testing VCO...")

# # Parameters
# feature_size_nm = 65
# V_dd = 1.8
# target_frequency = 1e9  # 10 MHz

# # Instantiate VCO
# vco = VCO(feature_size_nm=feature_size_nm, V_dd=V_dd, target_frequecny=target_frequency)

# # Display Results
# print(f"Feature size (nm): {vco.feature_size_nm}")
# print(f"V_dd: {vco.V_dd} V")
# print(f"Target Frequency: {vco.target_frequency} Hz")
# print(f"Inverter delay: {vco.t_inv:.3e} s")
# print(f"Number of inverter stages: {vco.num_inverter}")
# print(f"Total switched capacitance: {vco.total_switch_cap:.3e} F")
# print(f"Estimated Power Consumption: {vco.compute_power():.3e} W")

# #TODO: Inverter delay is 0.4ps, correct value: around 40ps