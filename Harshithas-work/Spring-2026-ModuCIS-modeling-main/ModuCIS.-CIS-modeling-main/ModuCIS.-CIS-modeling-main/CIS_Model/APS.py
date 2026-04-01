
from Photodiode import Photodiode
from parameter_class import NMOS, PMOS
from wire import Wire
import numpy as np
import matplotlib.pyplot as plt
from Noise import Shot_Noise, Dark_Current_Noise, Thermal_Noise, Transfer_Noise, Current_Shot_Noise, Reset_Noise
import math
    

# Active Pixel Sensor (APS) class
# Models an APS pixel with photodiode, source follower, reset, and select transistors
# 
# Inputs: pd_E (optical power density in W/m²), bias_current (bias current in A), exposure_time (exposure time in s),
#         num_of_tap (number of taps), num_PD_per_tap (number of photodiodes per tap), color (color filter index, RGB),
#         photodiode_type (0=normal, 1=perovskite, 2=Foveon), feature_size_nm (process technology node in nm),
#         pd_length (photodiode length in μm), pd_width (photodiode width in μm), V_dd (supply voltage in V),
#         V_swing (voltage swing in V), Pixel_type (pixel type index),
#         PD_tech (photodiode technology index), capm2 (capacitance per unit area in F/m², optional),
#         EQE (external quantum efficiency, optional), lambda_m (wavelength in m, optional)
# Outputs: pd_capacitance (photodiode capacitance in F), pd_current (photodiode current in A), rst_time (reset time in s),
#          exposure_time (exposure time in s), input_capacitance (input capacitance in F), rst_voltage (reset voltage in V),
#          and various noise models (shot_noise_model, dark_current_noise_model, etc.)
# The CLass APS is used for both 3T and 4T APS design.
class APS:
    def __init__(self, pd_E, bias_current, exposure_time, num_of_tap, num_PD_per_tap, color, photodiode_type, feature_size_nm, pd_length, pd_width, V_dd, V_swing, Pixel_type, PD_tech, capm2=None, EQE=None, lambda_m=None):
        self.bias_current = bias_current
        self.V_dd = V_dd

        #Initialize the reset transistor, it is used to reset the photodiode.
        self.rst_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        #Initialize the select transistor, it is used to output the signal to the readout circuit.
        self.sel_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        #Initialize the source follower transistor, it is used to amplify the signal from the photodiode.
        self.sf = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        #Initialize the transmission gate transistor, it is used to transfer the signal from the photodiode to the floating diffusion.
        #It is for the 4T APS design.
        #has the same size as rst transistor, only difference is the diffusion region area
        self.tg = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        #Assume the drain diffusion length 50 times longer to create a FD; FLoating diffusion is actually the drain of the transmission gate transistor.
        self.tg.drain_cap = self.tg.drain_cap * 50 
        #Calculate the floating diffusion capacitance, it is the capacitance of the parasitic capacitance of the transmission gate transistor and the source follower transistor.
        self.fd_cap = self.tg.drain_cap * num_PD_per_tap + self.sf.gate_cap
        #Intialize the photodiode, it is used to convert the optical signal to the electrical signal.
        self.PD = Photodiode(
            area_um2=pd_length*pd_width,
            tech_node_nm=feature_size_nm,
            E=pd_E,
            V_swing = V_swing,
            color=color,
            photodiode_type = photodiode_type,
            Pixel_type = Pixel_type,
            PD_tech = PD_tech,
            capm2=capm2,
            EQE=EQE,
            lambda_m=lambda_m
        )
        
        # Parameters for the APS pixels
        # the reset voltage of the FD
        self.rst_voltage = (self.V_dd - self.tg.v_th) * 0.99 ##when the rst voltage reach to 90% of the Vgs-Vth, stop reseting

        # the capacitance of the a single photodiode
        self.pd_capacitance = self.PD.capacitance

        # The total capacitance of the at the node of the photodiode and the transmission gate transistor.
        self.pd_node_cap = (self.pd_capacitance + self.tg.source_cap) * num_PD_per_tap

        # the photocurrent of the photodiode
        self.pd_current = self.PD.I_pd * num_PD_per_tap

        # the reset time of the photodiode
        self.rst_time = self.compute_reset_time(self.pd_node_cap + self.tg.source_cap + self.fd_cap) #
        # the reset time of the floating diffusion
        self.fd_rst_time = self.compute_reset_time(self.fd_cap) #TODO: here I assume the fd is charge from 0 to reset voltage, but infact, the fd isnot begin from 0
        # the maximum exposure time of the photodiode
        self.exposure_time = self.compute_exposure_time(self.rst_voltage)
        # if user has input, use user's input. else use calculated max exposure time
        if exposure_time != 0:
            self.exposure_time = exposure_time

        # the input capacitance of the reset transistor
        self.input_capacitance = self.rst_transistor.gate_cap

        # the external quantum efficiency of the photodiode
        self.pd_eta = self.PD.eta

        #Initialize the shot noise model, it is used to calculate the shot noise of the photodiode.
        self.shot_noise_model = Shot_Noise(
            area_um2=pd_length*pd_width,
            feature_size_nm=feature_size_nm,
            reverse_bias=self.V_dd-self.rst_transistor.v_th,
            lambda_m=self.PD.lambda_m,
            eta=self.pd_eta,
            optical_power=pd_E,
            exposure_time=self.exposure_time
        )

        #Initialize the dark current noise model, it is used to calculate the dark current noise of the photodiode.
        self.dark_current_noise_model = Dark_Current_Noise(
            area_um2=pd_length*pd_width,
            exposure_time=self.exposure_time,
            dark_currnet_density=14.75e-9 #TODO:modeling it
        )
        
        #initialize the source follower's Vds 
        self.sf_vds = 0 #update at the top class 
        #initialize the load capacitance of the source follower
        self.load_cap = 5e-12 #update at the top class
        #calculate the source follower's transconductance
        self.sf_transconductance = self.compute_SF_transconductance()

        #Initialize the thermal noise model, it is used to calculate the thermal noise of the source follower.
        self.sf_thermal_noise_model = Thermal_Noise(
            transconductance=self.sf_transconductance,
            cap=self.fd_cap,
            temperature=300
        )

        #calculate the transmission gate's transconductance
        self.tg_transconductance = self.compute_TG_transconductance()
        #Initialize the thermal noise model, it is used to calculate the thermal noise of the transmission gate.
        self.tg_thermal_noise_model = Thermal_Noise(
            transconductance=self.tg_transconductance,
            cap=self.pd_capacitance,
            temperature=300
        )

        #Initialize the transfer noise model, it is used to calculate the transfer noise of the photodiode.
        self.transfer_noise_model = Transfer_Noise(
            num_photoelectrons=self.shot_noise_model.shot_noise_square
        )

        #calculate the source follower's output resistance
        self.sf_output_resistance = 1/math.sqrt(2 * self.sf.un * self.sf.Cox * (self.sf.width / self.sf.length) * self.bias_current)

        #Initialize the current shot noise model, it is used to calculate the current shot noise of the source follower.
        self.current_shot_noise_model = Current_Shot_Noise(
            current=self.bias_current,
            transconductance=self.sf_transconductance,
            input_cap=self.fd_cap,
            load_cap=self.load_cap,
            output_res=self.sf_output_resistance
        )

        #Extract the noise square of the photodiode, source follower, transmission gate, and calcautle the total noise square.
        self.photo_shot_noise_square = self.shot_noise_model.shot_noise_square * num_PD_per_tap
        self.dark_current_noise_square = self.dark_current_noise_model.dark_current_noise_square * num_PD_per_tap
        self.sf_shot_noise_square = self.current_shot_noise_model.current_shot_noise_square
        self.transfer_noise_square = self.transfer_noise_model.transfer_noise_square
        self.sf_thermal_noise_square = self.sf_thermal_noise_model.thermal_noise_square
        self.tg_thermal_noise_square = self.tg_thermal_noise_model.thermal_noise_square
        self.total_thermal_noise_square = self.sf_thermal_noise_square + self.tg_thermal_noise_square

    #Calculate the source follower's transconductance
    def compute_SF_transconductance(self):
        gm =math.sqrt(2*self.sf.un*self.sf.Cox*(self.sf.width/self.sf.length)*self.bias_current)*(1+self.sf.lambda_param*self.sf_vds)
        return gm
    
    #Calculate the transmission gate's transconductance
    def compute_TG_transconductance(self):
        gm =math.sqrt(2*self.tg.un*self.tg.Cox*(self.tg.width/self.tg.length)*self.bias_current)
        return gm
    
    #Calculate the reset time
    def compute_reset_time(self, cap):
        #compute the reset time
        V_pd = 0
        for val in np.arange(1e-12, 1.0001e-8, 1e-12):
            Vx = self.V_dd - V_pd #Vds = Vgs
            Vth = self.rst_transistor.v_th #updated Vth due to body effect
            
            #because the Vgs always equal to Vds, the reset transistor are in Saturation all the time
            I = 0.5 * self.rst_transistor.un * self.rst_transistor.Cox * ((Vx - Vth) **2)*(1+self.rst_transistor.lambda_param*Vx)
            V_pd = V_pd + I*1e-12/cap
            if V_pd >= self.rst_voltage: #when the rst voltage reach to 90% of the Vgs-Vth, stop reseting
                return val
        return val

    #Calculate the maximum exposure time
    def compute_exposure_time(self, voltage):
        #compute the max exposure time
        T_exposure = (self.pd_node_cap) * (voltage) / self.pd_current
        return T_exposure 
    
    #Calculate the reset dynamic energy
    def compute_reset_dynamic_energy(self, PD, FD):
        #the energy during reset.
        PD_energy = (self.pd_capacitance+self.rst_transistor.drain_cap+self.tg.source_cap) * self.rst_voltage * self.V_dd
        FD_energy = self.fd_cap * self.rst_voltage * self.V_dd
        if PD and FD:
            energy = PD_energy + FD_energy
        elif PD:
            energy = PD_energy
        else:
            energy = FD_energy
        return energy


# Capacitive Transimpedance Amplifier (CTIA) class
# Models a CTIA pixel with feedback capacitor for charge integration
# Uses a feedback capacitor instead of floating diffusion for better linearity and noise performance
# 
# Inputs: pd_E (optical power density in W/m²), bias_current (bias current in A), exposure_time (exposure time in s),
#         color (color filter index, RGB), num_PD_per_tap (number of photodiodes per tap), photodiode_type (0=normal, 1=perovskite, 2=Foveon),
#         num_of_tap (number of taps), feature_size_nm (process technology node in nm), pd_length (photodiode length in μm),
#         pd_width (photodiode width in μm), V_dd (supply voltage in V), 
#         V_swing (voltage swing in V), Cap_FB (feedback capacitance in F), load_cap (load capacitance in F),
#         PD_tech (photodiode technology index), capm2 (capacitance per unit area in F/m², optional),
#         EQE (external quantum efficiency, optional), lambda_m (wavelength in m, optional)
# Outputs: pd_capacitance (photodiode capacitance in F), pd_current (photodiode current in A), rst_time (reset time in s),
#          exposure_time (exposure time in s), input_capacitance (input capacitance in F), rst_voltage (reset voltage in V),
#          static_power (static power consumption in W), dynamic_energy (dynamic energy per transition in J),
#          and various noise models (shot_noise_model, dark_current_noise_model, etc.)
class CTIA:
    def __init__(self, pd_E, bias_current, exposure_time,color,num_PD_per_tap, photodiode_type, num_of_tap, feature_size_nm, pd_length, pd_width, V_dd, V_swing , Cap_FB, load_cap, PD_tech, capm2=None, EQE=None, lambda_m=None):
        self.bias_current = bias_current
        self.V_dd = V_dd
        self.load_cap = load_cap
        self.rst_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.sel_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.sf = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        #has the same size as rst transistor, only difference is the diffusion region area
        self.tg = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.CTIA_bias_transistor = PMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*4,
            length=feature_size_nm,
            multiplier=1
        )

        #Initialize the total feedback capacitance, that include all the parasitic capacitance.
        self.Cap_FB = Cap_FB + self.CTIA_bias_transistor.drain_cap + self.tg.drain_cap*3 + self.tg.source_cap + self.tg.gate_cap

        #Calculate the bias current of the CTIA bias transistor.
        self.CTIA_bias_current = self.CTIA_bias_transistor.bias_current
        #Calculate the static power of the CTIA bias transistor.
        self.static_power = self.CTIA_bias_current*self.V_dd*num_of_tap
        #Calculate the dynamic energy per transition of the feedback cap.
        self.dynamic_energy = 1/2*self.Cap_FB*(V_dd**2)*num_of_tap

        self.area = pd_length*pd_width
        #Initialize the photodiode.
        self.PD = Photodiode(
            area_um2= pd_length*pd_width,
            tech_node_nm=feature_size_nm,
            E=pd_E,
            V_swing = V_swing,
            color=color,
            photodiode_type = photodiode_type,
            Pixel_type= 0,
            PD_tech = PD_tech,
            capm2=capm2,
            EQE=EQE,
            lambda_m=lambda_m
        )


        self.rst_voltage = (self.V_dd - self.CTIA_bias_transistor.v_th) * 0.9 ##when the rst voltage reach to 90% of the Vgs-Vth, stop reseting
        self.pd_capacitance = self.PD.capacitance
        self.pd_node_cap = self.pd_capacitance + self.tg.source_cap
        self.pd_current = self.PD.I_pd
        self.rst_time = self.compute_reset_time(self.rst_transistor.source_cap + self.rst_transistor.drain_cap + self.Cap_FB) #
        self.exposure_time = self.compute_exposure_time(self.rst_voltage)
        if exposure_time != 0:
            self.exposure_time = exposure_time
        print(self.exposure_time)
        self.input_capacitance = self.rst_transistor.gate_cap
        self.pd_eta = self.PD.eta

        #Initialize the shot noise model, it is used to calculate the shot noise of the photodiode.0
        self.shot_noise_model = Shot_Noise(
            area_um2=pd_length*pd_width,
            feature_size_nm=feature_size_nm,
            reverse_bias=self.V_dd-self.rst_transistor.v_th,
            lambda_m=self.PD.lambda_m,
            eta=self.pd_eta,
            optical_power=pd_E,
            exposure_time=self.exposure_time
        )
        
        self.dark_current_noise_model = Dark_Current_Noise(
            area_um2=pd_length*pd_width,
            exposure_time=self.exposure_time,
            dark_currnet_density=14.75e-9 #TODO:make it change able
        )

        self.sf_vds = 0 #update at APS array
        self.load_cap = 5e-12 #update at APS array
        self.sf_transconductance = self.compute_SF_transconductance()

        self.sf_thermal_noise_model = Thermal_Noise(
            transconductance=self.sf_transconductance,
            cap=self.Cap_FB,
            temperature=300
        )

        self.tg_transconductance = self.compute_TG_transconductance()
        self.tg_thermal_noise_model = Thermal_Noise(
            transconductance=self.tg_transconductance,
            cap=self.pd_capacitance,
            temperature=300
        )

        self.transfer_noise_model = Transfer_Noise(
            num_photoelectrons=self.shot_noise_model.shot_noise_square
        )

        self.sf_output_resistance = 1/math.sqrt(2 * self.sf.un * self.sf.Cox * (self.sf.width / self.sf.length) * self.bias_current)

        self.current_shot_noise_model = Current_Shot_Noise(
            current=self.bias_current,
            transconductance=self.sf_transconductance,
            input_cap=self.Cap_FB,
            load_cap=self.load_cap,
            output_res=self.sf_output_resistance
        )

        self.photo_shot_noise_square = self.shot_noise_model.shot_noise_square*num_PD_per_tap
        self.dark_current_noise_square = self.dark_current_noise_model.dark_current_noise_square*num_PD_per_tap
        self.sf_shot_noise_square = self.current_shot_noise_model.current_shot_noise_square
        self.transfer_noise_square = self.transfer_noise_model.transfer_noise_square
        self.sf_thermal_noise_square = self.sf_thermal_noise_model.thermal_noise_square
        self.tg_thermal_noise_square = self.tg_thermal_noise_model.thermal_noise_square
        self.total_thermal_noise_square = self.sf_thermal_noise_square + self.tg_thermal_noise_square


    #Calculate the source follower's transconductance
    def compute_SF_transconductance(self):
        gm =math.sqrt(2*self.sf.un*self.sf.Cox*(self.sf.width/self.sf.length)*self.bias_current)*(1+self.sf.lambda_param*self.sf_vds)
        return gm
    
    #Calculate the transmission gate's transconductance
    def compute_TG_transconductance(self):
        gm =math.sqrt(2*self.tg.un*self.tg.Cox*(self.tg.width/self.tg.length)*self.bias_current)
        return gm
    
    #Calculate the reset time
    def compute_reset_time(self, cap):
        R = self.rst_transistor.R_on
        C = cap
        val = R*C
        return val

    #Calculate the maximum exposure time
    def compute_exposure_time(self, voltage):
        #compute the max exposure time
        T_exposure = (self.pd_capacitance + self.tg.source_cap) * (voltage) / self.pd_current
        return T_exposure 
    

# CNN Photodiode (CNN_PD) class
# Models a photodiode structure optimized for CNN-based image sensors
# Used in CNN processing path for optical neural network applications
# 
# Inputs: pd_E (optical power density in W/m²), bias_current (bias current in A), exposure_time (exposure time in s),
#         num_PD_per_tap (number of photodiodes per tap), color (color filter index, RGB), photodiode_type (0=normal, 1=perovskite, 2=Foveon),
#         feature_size_nm (process technology node in nm), pd_length (photodiode length in μm), pd_width (photodiode width in μm),
#         V_dd (supply voltage in V), V_swing (voltage swing in V),
#         PD_tech (photodiode technology index), capm2 (capacitance per unit area in F/m², optional),
#         EQE (external quantum efficiency, optional), lambda_m (wavelength in m, optional)
# Outputs: pd_capacitance (photodiode capacitance in F), pd_current (photodiode current in A), rst_time (reset time in s),
#          exposure_time (exposure time in s), input_capacitance (input capacitance in F), rst_voltage (reset voltage in V),
#          fd_rst_time (floating diffusion reset time in s), and various noise models (shot_noise_model, dark_current_noise_model, etc.)
# For this one, the structure is nearly the same as the APS class. For this class, the top calss mainly unse the information about the photodiode and the noise.
# Other PMW specific circuits, like the PMW comaprater, are not included in this class. And is defined at the top class file.
class CNN_PD:
    def __init__(self, pd_E, bias_current, exposure_time, num_PD_per_tap, color, photodiode_type, feature_size_nm, pd_length, pd_width, V_dd, V_swing, PD_tech, capm2=None, EQE=None, lambda_m=None):
        self.bias_current = bias_current
        self.V_dd = V_dd
        self.rst_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.sel_transistor = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        self.sf = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        #has the same size as rst transistor, only difference is the diffusion region area
        self.tg = NMOS(
            tech_params=feature_size_nm,
            width=feature_size_nm*2,
            length=feature_size_nm,
            multiplier=1
        )

        #Assume the drain diffusion length 5 times longer to create a FD
        self.tg.drain_cap = self.tg.drain_cap * 50 #TODO find a better number for the FD cap 
        self.fd_cap = self.tg.drain_cap * num_PD_per_tap + self.sf.gate_cap
        self.PD = Photodiode(
            area_um2=pd_length*pd_width,
            tech_node_nm=feature_size_nm,
            E=pd_E,
            V_swing = V_swing,
            color=color,
            photodiode_type = photodiode_type,
            Pixel_type= 0,
            num_PD_per_tap = num_PD_per_tap,
            PD_tech = PD_tech,
            capm2=capm2,
            EQE=EQE,
            lambda_m=lambda_m
        )

        self.rst_voltage = (self.V_dd - self.tg.v_th) * 0.99 ##when the rst voltage reach to 90% of the Vgs-Vth, stop reseting
        self.pd_capacitance = self.PD.capacitance
        self.pd_node_cap = (self.pd_capacitance + self.tg.source_cap) * num_PD_per_tap
        self.pd_current = self.PD.I_pd
        self.rst_time = self.compute_reset_time(self.pd_node_cap + self.tg.source_cap + self.fd_cap) #
        self.fd_rst_time = self.compute_reset_time(self.fd_cap) #TODO: here I assume the fd is charge from 0 to reset voltage, but infact, the fd isnot begin from 0
        self.exposure_time = self.compute_exposure_time(self.rst_voltage)
        if exposure_time != 0:
            self.exposure_time = exposure_time
        self.input_capacitance = self.rst_transistor.gate_cap
        self.pd_eta = self.PD.eta
        #self.input_resistance = self.rst_transistor.gate_on_resistance

        #TODO: Make noise realted to PD number
        self.shot_noise_model = Shot_Noise(
            area_um2=pd_length*pd_width,
            feature_size_nm=feature_size_nm,
            reverse_bias=self.V_dd-self.rst_transistor.v_th,
            lambda_m=self.PD.lambda_m,
            eta=self.pd_eta,
            optical_power=pd_E,
            exposure_time=self.exposure_time
        )
        
        self.dark_current_noise_model = Dark_Current_Noise(
            area_um2=pd_length*pd_width,
            exposure_time=self.exposure_time,
            dark_currnet_density=14.75e-9 #TODO:make it change able
        )

        self.sf_vds = 0 #update at APS array
        self.load_cap = 5e-12 #update at APS array
        self.sf_transconductance = self.compute_SF_transconductance()

        self.sf_thermal_noise_model = Thermal_Noise(
            transconductance=self.sf_transconductance,
            cap=self.fd_cap,
            temperature=300
        )

        self.tg_transconductance = self.compute_TG_transconductance()
        self.tg_thermal_noise_model = Thermal_Noise(
            transconductance=self.tg_transconductance,
            cap=self.pd_capacitance,
            temperature=300
        )

        self.transfer_noise_model = Transfer_Noise(
            num_photoelectrons=self.shot_noise_model.shot_noise_square
        )

        self.sf_output_resistance = 1/math.sqrt(2 * self.sf.un * self.sf.Cox * (self.sf.width / self.sf.length) * self.bias_current)

        self.current_shot_noise_model = Current_Shot_Noise(
            current=self.bias_current,
            transconductance=self.sf_transconductance,
            input_cap=self.fd_cap,
            load_cap=self.load_cap,
            output_res=self.sf_output_resistance
        )

        self.photo_shot_noise_square = self.shot_noise_model.shot_noise_square * num_PD_per_tap
        self.dark_current_noise_square = self.dark_current_noise_model.dark_current_noise_square * num_PD_per_tap
        self.sf_shot_noise_square = self.current_shot_noise_model.current_shot_noise_square
        self.transfer_noise_square = self.transfer_noise_model.transfer_noise_square
        self.sf_thermal_noise_square = self.sf_thermal_noise_model.thermal_noise_square
        self.tg_thermal_noise_square = self.tg_thermal_noise_model.thermal_noise_square
        self.total_thermal_noise_square = self.sf_thermal_noise_square + self.tg_thermal_noise_square


    #Calculate the source follower's transconductance
    def compute_SF_transconductance(self):
        gm =math.sqrt(2*self.sf.un*self.sf.Cox*(self.sf.width/self.sf.length)*self.bias_current)*(1+self.sf.lambda_param*self.sf_vds)
        return gm
    
    #Calculate the transmission gate's transconductance
    def compute_TG_transconductance(self):
        gm =math.sqrt(2*self.tg.un*self.tg.Cox*(self.tg.width/self.tg.length)*self.bias_current)
        return gm
    
    #Calculate the reset time
    def compute_reset_time(self, cap):
        #compute the reset time
        V_pd = 0
        for val in np.arange(1e-12, 1.0001e-8, 1e-12):
            Vx = self.V_dd - V_pd #Vds = Vgs
            Vth = self.rst_transistor.v_th #updated Vth due to body effect
            
            #because the Vgs always equal to Vds, the reset transistor are in Saturation all the time
            I = 0.5 * self.rst_transistor.un * self.rst_transistor.Cox * ((Vx - Vth) **2)*(1+self.rst_transistor.lambda_param*Vx)
            V_pd = V_pd + I*1e-12/cap
            if V_pd >= self.rst_voltage: #when the rst voltage reach to 90% of the Vgs-Vth, stop reseting
                return val
        return val

    #Calculate the maximum exposure time
    def compute_exposure_time(self, voltage):
        #compute the max exposure time
        T_exposure = (self.pd_node_cap) * (voltage) / self.pd_current
        return T_exposure 
    
    #Calculate the reset dynamic energy
    def compute_reset_dynamic_energy(self, PD, FD):
        #the energy during reset.
        PD_energy = (self.pd_capacitance+self.rst_transistor.drain_cap+self.tg.source_cap) * self.rst_voltage * self.V_dd
        FD_energy = self.fd_cap * self.rst_voltage * self.V_dd
        if PD and FD:
            energy = PD_energy + FD_energy
        elif PD:
            energy = PD_energy
        else:
            energy = FD_energy
        return energy

FourTAPS = APS
