import numpy as np
import math

# Technology parameters class
# Stores technology-specific parameters for different process nodes (130nm, 65nm, 45nm)
# 
# Inputs: feature_size_nm (process technology node in nm, optional)
# Outputs: t_ox (oxide thickness in m), min_width (minimum transistor width in nm), max_width (maximum transistor width in nm),
#          V_th0 (zero-bias threshold voltage in V), body_effect_coefficient (body effect coefficient),
#          fermi_potential (Fermi potential in V), cap_ideal_gate (ideal gate capacitance per unit width in F/m),
#          cap_overlap (overlap capacitance per unit width in F/m), cap_fringe (fringe capacitance per unit width in F/m),
#          phy_gate_length (physical gate length in m), cap_junction (junction capacitance per unit area in F/m²),
#          cap_sidewall (sidewall capacitance per unit length in F/m), lambda_param (channel length modulation parameter)
class TechnologyParameters:
    
    def __init__(self, feature_size_nm=None):
        if feature_size_nm >= 130:
            self.t_ox = 2.2e-9
            self.min_width = 130
            self.max_width = 130*40
            self.V_th0 = 490e-3
            self.body_effect_coefficient = 0.45
            self.fermi_potential = 0.9
            self.cap_ideal_gate = 7.41e-10
            self.cap_overlap = 7.41e-10*0.2
            self.cap_fringe =2.4e-10
            self.phy_gate_length = 0.0451e-6
            self.cap_polywire = 0
            self.cap_junction = 1e-3
            self.cap_sidewall = 2.5e-10
            self.length_poly = 0
            self.lambda_param = 0.06
        elif feature_size_nm >= 65:
            self.t_ox = 1.8e-9
            self.min_width = 65
            self.max_width = 65*40
            self.V_th0 = 400e-3
            self.body_effect_coefficient = 0.45
            self.fermi_potential = 0.9
            self.cap_ideal_gate = 4.7e-10
            self.cap_overlap = 4.7e-10*0.2
            self.cap_fringe =2.4e-10
            self.phy_gate_length = 0.025e-6
            self.cap_polywire = 0
            self.cap_junction = 1e-3
            self.cap_sidewall = 2.5e-10
            self.length_poly = 0
            self.lambda_param = 0.06
        elif feature_size_nm >= 45:
            self.t_ox = 1.2e-9
            self.min_width = 45
            self.max_width = 45*40
            self.V_th0 = 400e-3
            self.body_effect_coefficient = 0.45
            self.fermi_potential = 0.9
            self.cap_ideal_gate = 6.78e-10
            self.cap_overlap = 6.78e-10*0.2
            self.cap_fringe =1.7e-10
            self.phy_gate_length = 0.018e-6
            self.cap_polywire = 0
            self.cap_junction = 1e-3
            self.cap_sidewall = 2.5e-10
            self.length_poly = 0
            self.lambda_param = 0.06
        else:
            raise ValueError("Feature size too large or unsupported.")
        
    

# NMOS transistor class
# Models an NMOS transistor with capacitance, transconductance, and body effect calculations
# 
# Inputs: tech_params (process technology node in nm), width (transistor width in nm, optional),
#         length (transistor length in nm, optional), multiplier (transistor multiplier, optional, default=1),
#         source_voltage (source voltage in V, optional, default=0)
# Outputs: Cox (oxide capacitance per unit area in F/m²), un (electron mobility in m²/V·s), v_th (threshold voltage in V),
#          gate_cap (gate capacitance in F), drain_cap (drain capacitance in F), source_cap (source capacitance in F),
#          lambda_param (channel length modulation parameter), gm (transconductance in S), R_on (on-resistance in Ω)
class NMOS:
    def __init__(self, tech_params, width=None, length=None, multiplier=1, source_voltage = 0):
        #permittivity of the gate oxide material, a constant
        self.eps_ox = 3.45e-11
        self.feature_size_nm = tech_params
        self.width = width #in nm
        self.length = length #in nm
        #multiplier of the transistor, used to scale the transistor size, 1 by default
        self.multiplier = multiplier
        self.source_voltage = source_voltage

        #bias current of the transistor, 1uA by default, this value is updated at the top class
        self.bias_current = 1e-6
        #drain-source voltage of the transistor, 0V by default, this value is updated at the top class
        self.DS_voltage = 0
        #technology parameters of the transistor, a constant
        self.tech = TechnologyParameters(
            feature_size_nm=self.feature_size_nm
        )

        self.Cox = self.eps_ox / self.tech.t_ox
        self.un = 350e-4
        self.v_th = self.compute_vth_with_body_effect()
        self.gate_cap = self.compute_gate_cap()
        self.drain_cap = self.compute_source_drain_cap()
        self.source_cap = self.compute_source_drain_cap()
        self.lambda_param = self.tech.lambda_param
        self.gm = self.compute_gm()
        self.R_on = 1/self.gm
        # Validate dimensions
        if self.width < self.tech.min_width:
            self.width = self.tech.min_width
            print(f"Warning: Width adjusted to minimum {tech_params.min_width}nm")
        if self.width > self.tech.max_width:
            self.width = self.tech.max_width
            print(f"Warning: Width adjusted to maximum {tech_params.max_width}nm")
        
    def compute_vth_with_body_effect(self):
        return self.tech.V_th0 + self.tech.body_effect_coefficient * (math.sqrt(abs(self.tech.fermi_potential + self.source_voltage)) - math.sqrt(abs(self.tech.fermi_potential)))
    
    # Equation from Destiny may not be very useful here
    # def compute_gate_cap(self):
    #     C_gate = (self.tech.cap_ideal_gate + self.tech.cap_overlap + 3*self.tech.cap_fringe) * self.width*1e-9 + self.tech.phy_gate_length * self.tech.cap_polywire
    #     return C_gate
    def compute_gm(self):
        gm =math.sqrt(2*self.un*self.Cox*(self.width/self.length)*self.bias_current)*(1+self.lambda_param*self.DS_voltage)
        return gm

    def compute_gate_cap(self):
        gate_cap = self.Cox * self.width*1e-9 * self.length*1e-9 + self.tech.cap_overlap * self.width*1e-9
        return gate_cap
    
    def compute_source_drain_cap(self):
        # A = W * L_D
        # P = 2 * (W + L_D)
        # C_ds = CJ * A + CJSW * P
        source_drain_cap = self.tech.cap_junction * self.width*1e-9 * (self.length*1e-9 + 2*self.tech.length_poly) + self.tech.cap_sidewall * 2 * (self.width*1e-9 + (self.length*1e-9 + 2*self.tech.length_poly))
        return source_drain_cap

# PMOS transistor class
# Models a PMOS transistor with capacitance, transconductance, and body effect calculations
# 
# Inputs: tech_params (process technology node in nm), width (transistor width in nm, optional),
#         length (transistor length in nm, optional), multiplier (transistor multiplier, optional, default=1),
#         source_voltage (source voltage in V, optional, default=0)
# Outputs: Cox (oxide capacitance per unit area in F/m²), un (hole mobility in m²/V·s), v_th (threshold voltage in V),
#          gate_cap (gate capacitance in F), drain_cap (drain capacitance in F), source_cap (source capacitance in F),
#          lambda_param (channel length modulation parameter), gm (transconductance in S), R_on (on-resistance in Ω)
class PMOS:
    def __init__(self, tech_params, width=None, length=None, multiplier=1, source_voltage = 0):
        #permittivity of the gate oxide material, a constant
        self.eps_ox = 3.45e-11
        self.feature_size_nm = tech_params
        self.width = width #in nm
        self.length = length #in nm
        #multiplier of the transistor, used to scale the transistor size, 1 by default
        self.multiplier = multiplier
        self.source_voltage = source_voltage
        #bias current of the transistor, 1uA by default, this value is updated at the top class
        self.bias_current = 1e-6
        #drain-source voltage of the transistor, 0V by default, this value is updated at the top class
        self.DS_voltage = 0
        #technology parameters of the transistor, a constant
        self.tech = TechnologyParameters(
            feature_size_nm=self.feature_size_nm
        )

        self.Cox = self.eps_ox / self.tech.t_ox
        self.un = 100e-4
        self.v_th = self.compute_vth_with_body_effect()
        self.gate_cap = self.compute_gate_cap()
        self.drain_cap = self.compute_source_drain_cap()
        self.source_cap = self.compute_source_drain_cap()
        self.lambda_param = self.tech.lambda_param
        self.gm = self.compute_gm()
        self.R_on = 1/self.gm
        
        # Validate dimensions
        if self.width < self.tech.min_width:
            self.width = self.tech.min_width
            print(f"Warning: Width adjusted to minimum {tech_params.min_width}nm")
        if self.width > self.tech.max_width:
            self.width = self.tech.max_width
            print(f"Warning: Width adjusted to maximum {tech_params.max_width}nm")
    
    def compute_gm(self):
        gm =math.sqrt(2*self.un*self.Cox*(self.width/self.length)*self.bias_current)*(1+self.lambda_param*self.DS_voltage)
        return gm

    def compute_vth_with_body_effect(self):
        return self.tech.V_th0 + self.tech.body_effect_coefficient * (math.sqrt(abs(self.tech.fermi_potential + self.source_voltage)) - math.sqrt(abs(self.tech.fermi_potential)))
    
    # Equation from Destiny may not be very useful here
    # def compute_gate_cap(self):
    #     C_gate = (self.tech.cap_ideal_gate + self.tech.cap_overlap + 3*self.tech.cap_fringe) * self.width*1e-9 + self.tech.phy_gate_length * self.tech.cap_polywire
    #     return C_gate

    def compute_gate_cap(self):
        gate_cap = self.Cox * self.width*1e-9 * self.length*1e-9 + self.tech.cap_overlap * self.width*1e-9
        return gate_cap
    
    def compute_source_drain_cap(self):
        # A = W * L_D
        # P = 2 * (W + L_D)
        # C_ds = CJ * A + CJSW * P
        source_drain_cap = self.tech.cap_junction * self.width*1e-9 * (self.length*1e-9 + 2*self.tech.length_poly) + self.tech.cap_sidewall * 2 * (self.width*1e-9 + (self.length*1e-9 + 2*self.tech.length_poly))
        return source_drain_cap
