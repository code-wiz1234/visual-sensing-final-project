# Wire class
# Models on-chip interconnect wires with resistance and capacitance calculations
# 
# Inputs: wire_type (wire type: local_aggressive, local_conservative, semi_aggressive, semi_conservative, global_aggressive, global_conservative),
#         feature_size_nm (process technology node in nm), temperature (temperature in K, optional, default=293)
# Outputs: res_wire_per_m (wire resistance per meter in Ω/m), cap_wire_per_m (wire capacitance per meter in F/m),
#          wire_width (wire width in m), wire_thickness (wire thickness in m), wire_spacing (wire spacing in m)
class Wire:
    # Constants
    #TODO: Cacti 6.0, wire RC model
    COPPER_RESISTIVITY_TEMP_COEFF = 0.0039  # 1/°C
    EPSILON_0 = 8.854e-12  # Vacuum permittivity (F/m)

    def __init__(self, wire_type: str, feature_size_nm: float, temperature=293):
        self.wire_type = wire_type
        self.feature_size_nm = feature_size_nm
        self.feature_size = feature_size_nm * 1e-9  # in meters
        self.temperature = temperature  # in Kelvin
        self.res_wire_per_m = 0
        self.cap_wire_per_m = 0


        self.set_wire_parameters()
        self.calculate_geometry()
        self.calculate_rc()

    def set_wire_parameters(self):
        wt = self.wire_type
        fs = self.feature_size

        if self.feature_size_nm <= 180:
            if wt == "local_aggressive":
                self.barrier_thickness = 0.00e-6
                self.horizontal_dielectric = 2.303
                self.wire_pitch = 2.5 * fs
                self.aspect_ratio = 2.7
                self.ild_thickness = 0.405e-6

            elif wt == "local_conservative":
                self.barrier_thickness = 0.006e-6
                self.horizontal_dielectric = 2.734
                self.wire_pitch = 2.5 * fs
                self.aspect_ratio = 2.0
                self.ild_thickness = 0.405e-6

            elif wt == "semi_aggressive":
                self.barrier_thickness = 0.00e-6
                self.horizontal_dielectric = 2.303
                self.wire_pitch = 4.0 * fs
                self.aspect_ratio = 2.7
                self.ild_thickness = 0.405e-6

            #for small size pixel array's input and output bus
            elif wt == "semi_conservative":
                self.barrier_thickness = 0.006e-6
                self.horizontal_dielectric = 2.734
                self.wire_pitch = 4.0 * fs
                self.aspect_ratio = 2.0
                self.ild_thickness = 0.405e-6

            elif wt == "global_aggressive":
                self.barrier_thickness = 0.00e-6
                self.horizontal_dielectric = 2.303
                self.wire_pitch = 8.0 * fs
                self.aspect_ratio = 2.8
                self.ild_thickness = 0.81e-6

            #for large size pixel array's input and output bus
            elif wt == "global_conservative":
                self.barrier_thickness = 0.006e-6
                self.horizontal_dielectric = 2.734
                self.wire_pitch = 8.0 * fs
                self.aspect_ratio = 2.2
                self.ild_thickness = 0.77e-6

            else:  # dram_wordline
                self.barrier_thickness = 0.0
                self.horizontal_dielectric = 0.0
                self.wire_pitch = 2.0 * fs
                self.aspect_ratio = 0.0
                self.ild_thickness = 0.0

        else:
            raise ValueError("Feature size too large or unsupported.")

    def calculate_geometry(self):
        self.wire_width = self.wire_pitch / 2
        self.wire_thickness = self.aspect_ratio * self.wire_width
        self.wire_spacing = self.wire_pitch - self.wire_width

    def calculate_rc(self):
        # Copper base resistivity and temp correction
        base_resistivity = 6.0e-8 if 'global' not in self.wire_type else 3.0e-8
        self.copper_resistivity = base_resistivity * (
            1 + self.COPPER_RESISTIVITY_TEMP_COEFF * (self.temperature - 293)
        )

        self.res_wire_per_m = self.calculate_wire_resistance(
            self.copper_resistivity, self.wire_width, self.wire_thickness,
            self.barrier_thickness, dishing_thickness=0.0, alpha_scatter=1.0
        )

        self.cap_wire_per_m = self.calculate_wire_capacitance(
            permittivity=self.EPSILON_0,
            wire_width=self.wire_width,
            wire_thickness=self.wire_thickness,
            wire_spacing=self.wire_spacing,
            ild_thickness=self.ild_thickness,
            miller_value=1.5,
            horizontal_dielectric=self.horizontal_dielectric,
            vertical_dielectric=3.9,
            fringe_cap=1.15e-10
        )
    
    #calculate the wire delay and enery; for energy, used for both input and output bus; for delay only used for input bus;
    #for output bus, beucase of the amplifier effect, the delay model cannot be used
    # def calculate_delay_energy(self, wire_length, vdd):
    #     R = self.res_wire_per_m
    #     C = self.cap_wire_per_m
    #     L = wire_length
    #     delay = 2.3 * R * C * (L ** 2) / 2
    #     energy = C * L * (vdd ** 2)
    #     leakage_power = 0.0
    #     return delay, energy, leakage_power

    @staticmethod
    #calculate the wire resistance per meter
    def calculate_wire_resistance(resistivity, wire_width, wire_thickness, barrier_thickness, dishing_thickness, alpha_scatter):
        effective_thickness = wire_thickness - barrier_thickness - dishing_thickness
        effective_width = wire_width - 2 * barrier_thickness
        return alpha_scatter * resistivity / (effective_thickness * effective_width)

    @staticmethod
    #calculate the wire capacitance per meter
    def calculate_wire_capacitance(permittivity, wire_width, wire_thickness, wire_spacing,
                                    ild_thickness, miller_value, horizontal_dielectric,
                                    vertical_dielectric, fringe_cap):
        vertical_cap = 2 * permittivity * vertical_dielectric * wire_width / ild_thickness
        sidewall_cap = 2 * permittivity * miller_value * horizontal_dielectric * wire_thickness / wire_spacing
        return vertical_cap + sidewall_cap + fringe_cap
