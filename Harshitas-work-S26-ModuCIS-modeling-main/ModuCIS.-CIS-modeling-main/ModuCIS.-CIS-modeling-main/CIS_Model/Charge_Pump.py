from parameter_class import NMOS, PMOS

# Charge pump class
# Models a charge pump circuit used in PLL for generating control voltage
# 
# Inputs:
#   Voltage Parameters:
#     - bias_voltage (bias voltage in V, optional, default=1.8)
#     - V_dd (supply voltage in V, optional, default=1.8)
#   Capacitance Parameters:
#     - load_cap (load capacitance in F)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm, optional)
# Outputs: bias_current (bias current in A), output_cap (output capacitance in F), total_input_cap (total input capacitance in F)
class Charge_Pump:
    def __init__(self, load_cap, feature_size_nm, V_dd, bias_voltage):
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd
        self.load_cap = load_cap
        self.bias_voltage = bias_voltage

        self.bias_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.switch_pull_up = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.switch_pull_down = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )
        self.bias_current = self.compute_bias_current()
        self.output_cap = self.switch_pull_down.drain_cap + self.switch_pull_up.source_cap
        self.total_input_cap = self.switch_pull_down.gate_cap + self.switch_pull_up.gate_cap

    def compute_bias_current(self):
        bias_current = 0.5 * self.bias_transistor.un * self.bias_transistor.Cox * ((self.bias_voltage - self.bias_transistor.v_th) **2)*(1+self.bias_transistor.lambda_param*self.bias_voltage)
        return bias_current