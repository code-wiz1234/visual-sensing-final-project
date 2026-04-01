# Photodiode class
# Models a photodiode sensor element with capacitance, photocurrent, and full well capacity calculations
# Supports different photodiode types: normal, perovskite, and Foveon
# 
# Inputs: area_um2 (photodiode area in μm²), tech_node_nm (process technology node in nm),
#         E (optical power density in W/m²), V_swing (voltage swing in V),
#         color (color filter index), photodiode_type (0=normal, 1=perovskite, 2=Foveon), Pixel_type (pixel type index),
#         PD_tech (photodiode technology index), num_PD_per_tap (number of photodiodes per tap, optional, default=1),
#         capm2 (capacitance per unit area in F/m², optional), EQE (external quantum efficiency, optional),
#         lambda_m (wavelength in m, optional)
# Outputs: capacitance (photodiode capacitance in F), area_m2 (photodiode area in m²), I_pd (photocurrent in A),
#          FWC (full well capacity in electrons), unit_cap (capacitance per unit area in F/m²),
#          eta (quantum efficiency), lambda_m (wavelength in m)
class Photodiode:
    # Optical Constants for the photodiode
    EPSILON_0 = 8.854e-12  # F/m
    EPSILON_SI = 11.7 * EPSILON_0  # Permittivity of silicon (F/m)
    Q = 1.602e-19  # Electron charge (C)
    H = 6.626e-34        # Planck's constant (J·s)
    C = 3e8              # Speed of light (m/s)
    def __init__(self, area_um2, tech_node_nm, E, V_swing, color, photodiode_type, Pixel_type, PD_tech, num_PD_per_tap=1, capm2=None, EQE=None, lambda_m=None):
        self.tech_node_nm = tech_node_nm

        # The build-in voltage of the photodiode, it is a PD related constant. Can be set as an user input.
        self.v_bi = 0.7  # Default built-in voltage
        self.E = E #input power density per unit area
        self.V_Swing = V_swing #the voltage swing of the photodiode, it is the voltage level when the photodiode is saturated.
        self.area_um2 = area_um2 #the area of the photodiode in μm²
        
        self.photodiode_type = photodiode_type #0 for normal photodiode, 1 for perovskite, 2 for Foveon
        self.Pixel_type = Pixel_type #Pixel_type (pixel type index: 0=4T_APS, 1=CTIA, 2=CNN_PD, 3=CNN_PD with CNN, 4=3T_APS)


        #Below are the default parameters for different photodiode types.
        self.CFA = CFA_array(color=color, Pixel_type = self.Pixel_type, PD_tech = PD_tech)
        self.perovskite = perovskite(color=color)
        self.Foven = Foveon(color=color)
        self.eta_all = [self.CFA.EQE, self.perovskite.EQE, self.Foven.EQE]
        self.lambda_all = [self.CFA.lambda_m, self.perovskite.lambda_m, self.Foven.lambda_m]
        self.unit_cap_all = [self.CFA.cap_m2, self.perovskite.cap_m2, self.Foven.cap_m2]
        
        #Set the unit capacitance, quantum efficiency, and wavelength for the photodiode.
        #If the user does not provide the unit capacitance, quantum efficiency, and wavelength, use the default parameters.
        self.unit_cap = capm2 if capm2 is not None else self.unit_cap_all[photodiode_type]
        self.eta = EQE if EQE is not None else self.eta_all[photodiode_type]
        self.lambda_m = lambda_m if lambda_m is not None else self.lambda_all[photodiode_type]
        
        #Compute the capacitance of the photodiode.
        self.capacitance = self.compute_capacitance()
        #Compute the area of the photodiode in m².
        self.area_m2 = self.area_um2 * 1e-12
        self.num_PD_per_tap = num_PD_per_tap #the number of photodiodes per tap.
        #Compute the photocurrent of the photodiode.
        self.I_pd = self.compute_photocurrent()*num_PD_per_tap
        #Compute the full well capacity of the photodiode.
        self.FWC = self.compute_FWC()

    #Compute the capacitance of the photodiode.
    def compute_capacitance(self):
        #calcaulte the capacitance
        Cap = self.unit_cap * self.area_um2 * 1e-12
        #Cap = self.FWC*self.Q / self.V_Swing
        return Cap
    
    #Compute the full well capacity of the photodiode.
    def compute_FWC(self):
        FWC = self.capacitance * self.V_Swing / self.Q
        return FWC

    #Compute the photocurrent of the photodiode.
    def compute_photocurrent(self):
        I_pd = self.eta * self.E * self.area_m2 * (self.Q * self.lambda_m) / (self.H * self.C)
        return I_pd
    
    #Set the unit capacitance of the photodiode.
    def set_capm2(self, capm2):
        self.unit_cap = capm2
        self.capacitance = self.compute_capacitance()
        self.FWC = self.compute_FWC()
    
    #Set the quantum efficiency of the photodiode.
    def set_EQE(self, EQE):
        self.eta = EQE
        self.I_pd = self.compute_photocurrent() * self.num_PD_per_tap
    
    #Set the wavelength of the photodiode.
    def set_lambda_m(self, lambda_m):
        self.lambda_m = lambda_m
        self.I_pd = self.compute_photocurrent() * self.num_PD_per_tap
    
    
# Perovskite photodiode class
# Models perovskite-based photodiode with color-specific parameters (EQE, capacitance, wavelength)
# 
# Inputs: color (color filter index: 0=R, 1=G, 2=B)
# Outputs: cap_m2 (capacitance per unit area in F/m²), EQE (external quantum efficiency), lambda_m (wavelength in m)
# This class include some default parameters for the perovskite photodiode
class perovskite:
    def __init__(self, color):
        #0 for Red; 1 for Green; 2 for Blue
        self.color = color
        self.cap_m2_all_color = [1.25e-3, 0.42e-3, 0.58e-3, 0, 0, 0, 0, 0, 0]
        self.EQE_all_color = [0.5, 0.4, 0.47, 0, 0, 0, 0, 0, 0]
        self.lambda_all_color = [630e-9, 550e-9, 450e-9, 0, 0, 0, 0, 0, 0]
        self.cap_m2 = self.cap_m2_all_color[color]
        self.EQE = self.EQE_all_color[color]
        self.lambda_m = self.lambda_all_color[color]
        
        

# Color Filter Array (CFA) class
# Models color filter array parameters for different colors and pixel types
# 
# Inputs: color (color filter index), Pixel_type (pixel type index), PD_tech (photodiode technology index)
# Outputs: EQE (external quantum efficiency), lambda_m (wavelength in m), cap_m2 (capacitance per unit area in F/m²)
# This class include some default parameters for the silicon photodiode with different color filters.
class CFA_array:
    def __init__(self, color, Pixel_type, PD_tech):
        self.color = color
        self.EQE_all_color = [0.13, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6,0.86]
        self.lambda_all_color = [630e-9, 540e-9, 460e-9, 510e-9, 500e-9, 500e-9, 500e-9, 500e-9,500e-9, 550e-9,]
        self.cap_m2 = 10e-4
        if Pixel_type == 4:
            self.cap_m2 *= 10
        self.EQE = self.EQE_all_color[color]
        self.lambda_m = self.lambda_all_color[color]

# Foveon photodiode class
# Models Foveon-type stacked photodiode with color-specific parameters
# 
# Inputs: color (color filter index: 0=Red, 1=Green, 2=Blue)
# Outputs: cap_m2 (capacitance per unit area in F/m²), EQE (external quantum efficiency), lambda_m (wavelength in m)
# This class include some default parameters for the Foveon photodiode
class Foveon:
    def __init__(self, color):
        self.color = color
        # unknown TODO: add cap for each color
        self.cap_m2_all_color = [15e-4, 12e-4, 10e-4, 0, 0, 0, 0, 0, 0]
        self.EQE_all_color = [0.13, 0.3, 0.1, 0, 0, 0, 0, 0, 0]
        self.lambda_all_color = [630e-9, 550e-9, 480e-9, 0, 0, 0, 0, 0, 0]
        self.cap_m2 = self.cap_m2_all_color[color]
        self.EQE = self.EQE_all_color[color]
        self.lambda_m = self.lambda_all_color[color]