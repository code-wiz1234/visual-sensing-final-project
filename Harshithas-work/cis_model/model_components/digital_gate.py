from parameter_class import NMOS, PMOS

# Inverter (INV) class
# Models a CMOS inverter gate with delay and capacitance calculations
# 
# Inputs: feature_size_nm (process technology node in nm), V_dd (supply voltage in V)
# Outputs: input_cap (input capacitance in F),
#          output_cap (output capacitance in F), resistance (equivalent pull-down resistance in Ω),
#          delay (inverter delay in s), total_switch_cap (total switching capacitance in F)
# A basic inverter that is made up of NMOS and PMOS transistors. 
class INV:
    def __init__(self, feature_size_nm, V_dd):
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd
        self.nmos_width = feature_size_nm*20
        self.nmos_length = feature_size_nm
        self.pmos_width = self.nmos_width*2
        self.pmos_length = feature_size_nm

        self.Pull_down = NMOS(
            tech_params=feature_size_nm,
            width=self.nmos_width,
            length=self.nmos_length,
            multiplier=1
        )

        self.Pull_up = PMOS(
            tech_params=feature_size_nm,
            width=self.pmos_width,
            length=self.pmos_length,
            multiplier=1
        )

        #Compute the input and output capacitance of the inverter.
        self.input_cap = self.Pull_down.gate_cap + self.Pull_up.gate_cap
        #Compute the output capacitance of the inverter.
        self.output_cap = self.Pull_down.drain_cap + self.Pull_up.drain_cap
        #Compute the resistance of the inverter.
        self.resistance = 1/(self.Pull_down.un*self.Pull_down.Cox*(self.Pull_down.width/self.Pull_down.length)*(self.V_dd-self.Pull_down.v_th))
        #Compute the delay of the inverter.
        self.delay = self.compute_delay()
        #Compute the total switching capacitance of the inverter.
        self.total_switch_cap = self.input_cap + self.output_cap

    def compute_delay(self):
        #other way to estimate the delay = 0.69*0.75*ClVdd/Idsatn
        #approximation of the inverter delay
        Beta = self.Pull_down.un*self.Pull_down.Cox*self.Pull_down.width/self.Pull_down.length
        #TODO: Capacitance is too low here
        Capacitance = self.input_cap+self.output_cap
        #I assume the controlled voltage is around Vdd
        t_inv = Capacitance/(Beta*(self.V_dd-self.Pull_down.v_th))
        return t_inv

# NOR gate class
# Models a CMOS NOR gate with input/output capacitance calculations
# 
# Inputs: feature_size_nm (process technology node in nm), V_dd (supply voltage in V)
# Outputs: input_cap_0 (input capacitance for input 0 in F),
#          input_cap_1 (input capacitance for input 1 in F), output_cap (output capacitance in F),
#          total_switch_cap (total switching capacitance in F)
# A basic two inputs NOR gate that is made up of NMOS and PMOS transistors. 
class NOR:
    def __init__(self, feature_size_nm, V_dd):
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd
        self.width = feature_size_nm*20
        self.length = feature_size_nm

        self.Pull_down_0 = NMOS(
            tech_params=feature_size_nm,
            width=self.width,
            length=self.length,
            multiplier=1
        )

        self.Pull_down_1 = NMOS(
            tech_params=feature_size_nm,
            width=self.width,
            length=self.length,
            multiplier=1
        )

        self.Pull_up_0 = PMOS(
            tech_params=feature_size_nm,
            width=self.width*2,
            length=self.length,
            multiplier=1
        )

        self.Pull_up_1 = PMOS(
            tech_params=feature_size_nm,
            width=self.width*2,
            length=self.length,
            multiplier=1
        )

        #Compute the input and output capacitance of the NOR gate. 
        self.input_cap_0 = self.Pull_down_0.gate_cap + self.Pull_up_0.gate_cap
        self.input_cap_1 = self.Pull_down_1.gate_cap + self.Pull_up_1.gate_cap
        self.output_cap = self.Pull_down_0.drain_cap + self.Pull_up_0.drain_cap + self.Pull_up_1.drain_cap
        #Compute the total switching capacitance of the NOR gate.
        self.total_switch_cap = self.input_cap_0 + self.input_cap_1 + self.output_cap

# NAND gate class
# Models a CMOS NAND gate with input/output capacitance calculations
# 
# Inputs: feature_size_nm (process technology node in nm), V_dd (supply voltage in V)
# Outputs: input_cap_0 (input capacitance for input 0 in F),
#          input_cap_1 (input capacitance for input 1 in F), output_cap (output capacitance in F),
#          total_switch_cap (total switching capacitance in F)
# A basic two inputs NAND gate that is made up of NMOS and PMOS transistors. 
class NAND:
    def __init__(self, feature_size_nm, V_dd):
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd
        self.width = feature_size_nm*20
        self.length = feature_size_nm

        self.Pull_down_0 = NMOS(
            tech_params=feature_size_nm,
            width=self.width,
            length=self.length,
            multiplier=1
        )

        self.Pull_down_1 = NMOS(
            tech_params=feature_size_nm,
            width=self.width,
            length=self.length,
            multiplier=1
        )

        self.Pull_up_0 = PMOS(
            tech_params=feature_size_nm,
            width=self.width*2,
            length=self.length,
            multiplier=1
        )

        self.Pull_up_1 = PMOS(
            tech_params=feature_size_nm,
            width=self.width*2,
            length=self.length,
            multiplier=1
        )

        #Compute the input and output capacitance of the NAND gate. 
        self.input_cap_0 = self.Pull_down_0.gate_cap + self.Pull_up_0.gate_cap
        self.input_cap_1 = self.Pull_down_1.gate_cap + self.Pull_up_1.gate_cap
        self.output_cap = self.Pull_down_0.drain_cap + self.Pull_up_0.drain_cap + self.Pull_up_1.drain_cap
        #Compute the total switching capacitance of the NAND gate.
        self.total_switch_cap = self.input_cap_0 + self.input_cap_1 + self.output_cap
# print("====== CMOS Inverter Characterization ======")

# # Example parameters
# feature_size_nm = 65
# V_dd = 1.8

# # Instantiate inverter
# inv = INV(feature_size_nm=feature_size_nm, V_dd=V_dd)

# print(f"Feature size: {inv.feature_size_nm} nm")
# print(f"Supply voltage: {inv.V_dd} V")

# print("\n--- NMOS/PMOS Parameters ---")
# print(f"NMOS W/L: {inv.Pull_down.width}/{inv.Pull_down.length} µm")
# print(f"PMOS W/L: {inv.Pull_up.width}/{inv.Pull_up.length} µm")

# print("\n--- Capacitance ---")
# print(f"Input Capacitance (C_in): {inv.input_cap:.3e} F")
# print(f"Output Capacitance (C_out): {inv.output_cap:.3e} F")

# print("\n--- Resistance and Delay ---")
# print(f"Equivalent Pull-down Resistance: {inv.resistance:.3e} Ω")
# print(f"Inverter Delay (t_inv): {inv.delay*1e12:.2f} ps")

# print("=============================================")