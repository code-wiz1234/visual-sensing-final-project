# Mobile Industry Processor Interface (MIPI) class
# Models MIPI interface for data transmission in CIS systems
# 
# Inputs:
#   Data Parameters:
#     - data_bandwidth (data bandwidth in bytes/s)
#   Timing Parameters:
#     - frame_rate (frame rate in fps)
#   Capacitance Parameters:
#     - MIPI_unit_cap (MIPI unit capacitance in F)
#   Process Parameters:
#     - feature_size_nm (process technology node in nm)
#   Power Parameters:
#     - MIPI_unit_energy (energy per byte in J, optional, default=100e-12)
# Outputs: MIPI_power (MIPI power consumption in W), MIPI_cap (total MIPI capacitance in F)
class MIPI:
    def __init__ (self, data_bandwidth, frame_rate, MIPI_unit_cap, feature_size_nm, MIPI_unit_energy = 100e-12):
        self.feature_size_nm = feature_size_nm

        #energy per byte
        self.MIPI_unit_energy = MIPI_unit_energy
        self.data_bandwidth = data_bandwidth
        self.frame_rate = frame_rate
        self.MIPI_power = self.compute_MIPI_power()
        self.MIPI_cap = MIPI_unit_cap*data_bandwidth
        print("MIPI", self.data_bandwidth, self.frame_rate)

    
    def compute_MIPI_power(self):
        power = self.MIPI_unit_energy*self.data_bandwidth*self.frame_rate
        return power

