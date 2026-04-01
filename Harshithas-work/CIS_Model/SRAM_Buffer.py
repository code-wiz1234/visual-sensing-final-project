from parameter_class import NMOS, PMOS
from digital_gate import INV
from wire import Wire
from Counter import Countner
from digital_gate import NAND
import math

# SRAM cell class
# Models a 6-transistor SRAM cell with bitline and wordline capacitance
# 
# Inputs: feature_size_nm (process technology node in nm, optional, default=65), V_dd (supply voltage in V, optional, default=1.8)
# Outputs: BL_cap (bitline capacitance in F),
#          BL_bar_cap (bitline-bar capacitance in F), WL_cap (wordline capacitance in F), CC_cap (cross-coupled capacitance in F)
# SRAM buffer based on the Destiny SRAM cells. Not used at the moment.
class SRAM_cell:
    def __init__(self, feature_size_nm = 65,V_dd = 1.8):
        self.V_dd = V_dd
        self.feature_size_nm = feature_size_nm

        self.switch_1 = NMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*20,
            length=self.feature_size_nm,
            multiplier=1
        )

        self.switch_2 = NMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*20,
            length=self.feature_size_nm,
            multiplier=1
        )

        self.inverter_1 = INV(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.inverter_2 = INV(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.inverter_1.nmos_width=feature_size_nm*20
        self.inverter_2.nmos_width=feature_size_nm*20

        self.BL_cap = self.switch_1.drain_cap
        self.BL_bar_cap = self.switch_2.drain_cap
        self.WL_cap = self.switch_1.gate_cap+self.switch_2.gate_cap
        self.CC_cap = self.inverter_1.input_cap + self.inverter_2.input_cap + self.inverter_1.output_cap + self.inverter_2.output_cap

# Multiplexer (MUX) class
# Models a multiplexer circuit for selecting data paths in SRAM buffer
# 
# Inputs: num_input (number of inputs), next_stage_input_Cap (next stage input capacitance in F),
#         feature_size_nm (process technology node in nm, optional, default=65), V_dd (supply voltage in V, optional, default=1.8)
# Outputs: capNMOSPassTransistor (NMOS pass transistor capacitance in F),
#          capOutput (output capacitance in F), resNMOSPassTransistor (NMOS pass transistor resistance in Ω),
#          readDynamicEnergy (read dynamic energy in J), writeDynamicEnergy (write dynamic energy in J)
class MUX:
    def __init__(self, num_input, next_stage_input_Cap, feature_size_nm = 65,V_dd = 1.8):
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd
        self.num_input = num_input

        self.nmos = NMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*20,
            length=self.feature_size_nm,
            multiplier=1
        )

        if self.num_input > 1:
            self.capNMOSPassTransistor = self.nmos.drain_cap
            self.capOutput = self.num_input * self.capNMOSPassTransistor + next_stage_input_Cap
            self.resNMOSPassTransistor = 1/self.nmos.gm
        else:
            self.capNMOSPassTransistor = 0
            self.capOutput = 0
            self.resNMOSPassTransistor = 0
        self.readDynamicEnergy = self.capOutput * self.V_dd * (self.V_dd - self.nmos.v_th)
        self.writeDynamicEnergy = self.readDynamicEnergy

# Decoder class
# Models a decoder circuit for address decoding in SRAM buffer
# 
# Inputs: num_input (number of input address bits), next_stage_input_Cap (next stage input capacitance in F, optional, default=0),
#         feature_size_nm (process technology node in nm, optional, default=65), V_dd (supply voltage in V, optional, default=1.8)
# Outputs: capOutput (output capacitance in F),
#          readDynamicEnergy (read dynamic energy in J), writeDynamicEnergy (write dynamic energy in J)
class Decoder:
    def __init__(self, num_input, next_stage_input_Cap = 0, feature_size_nm = 65,V_dd = 1.8):
        self.feature_size_nm = feature_size_nm
        self.V_dd = V_dd
        self.num_input = num_input

        self.nand = NAND(
            feature_size_nm=feature_size_nm,
            V_dd=self.V_dd
        )

        self.nand.width = feature_size_nm*20

        self.calculate_rc(next_stage_input_Cap)
        self.calculate_power()

    def calculate_rc(self, next_stage_input_Cap):
        if self.num_input > 1:
            self.capOutput = self.num_input * self.nand.output_cap + next_stage_input_Cap
    
    def calculate_power(self):
        # Dynamic Energy Calculation (Read/Write)
        #multiply 2, because 
        self.readDynamicEnergy = self.capOutput * (self.V_dd**2)
        self.writeDynamicEnergy = self.readDynamicEnergy

# Sense amplifier class
# Models a sense amplifier circuit for reading data from SRAM bitlines
# 
# Inputs: feature_size_nm (process technology node in nm, optional, default=65), V_dd (supply voltage in V, optional, default=1.8),
#         sense_bias_voltage (sense amplifier bias voltage in V, optional, default=0.6)
# Outputs: BL_cap (bitline capacitance in F),
#          BL_bar_cap (bitline-bar capacitance in F), output_cap (output capacitance in F), internal_cap (internal capacitance in F)
class Sense_Amp:
    def __init__(self, feature_size_nm = 65,V_dd = 1.8, sense_bias_voltage = 0.6):
        self.V_dd = V_dd
        self.feature_size_nm = feature_size_nm

        self.input_transistor1 = NMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*2,
            length=self.feature_size_nm,
            multiplier=1
        )

        self.input_transistor2 = NMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*2,
            length=self.feature_size_nm,
            multiplier=1
        )

        self.pull_up = PMOS(
            tech_params=feature_size_nm,
            width=self.feature_size_nm*4,
            length=self.feature_size_nm,
            multiplier=1
        )

        self.BL_cap = self.input_transistor1.drain_cap
        self.BL_bar_cap = self.input_transistor2.drain_cap
        self.output_cap = self.input_transistor2.drain_cap+self.pull_up.drain_cap
        self.internal_cap = self.input_transistor2.drain_cap+self.pull_up.drain_cap+self.pull_up.gate_cap*2

# SRAM buffer class
# Models a complete SRAM buffer system with cells, sense amplifiers, MUX, decoder, and counters
# 
# Inputs: frame_rate (frame rate in fps), next_stage_input_cap (next stage input capacitance in F), input_clk_freq (input clock frequency in Hz),
#         num_cols (number of columns), size (SRAM size: ADC_resolution × num_cols × 2), feature_size_nm (process technology node in nm, optional, default=65),
#         V_dd (supply voltage in V, optional, default=1.8)
# Outputs: read_power (read power in W), write_power (write power in W), bus_power (bus power in W), counter_power (counter power in W),
#          mux_power (multiplexer power in W), decoder_power (decoder power in W), total_power (total SRAM buffer power in W)
class SRAM:
    def __init__(self, frame_rate, next_stage_input_cap, input_clk_freq, num_cols, size, feature_size_nm = 65,V_dd = 1.8):
        self.input_clk_freq = input_clk_freq #should be the same the Input Driver, do one read/wirte every readout time
        self.num_cols = num_cols
        self.V_dd = V_dd
        self.feature_size_nm = feature_size_nm
        self.size = size #it should ADC resolution X num of cols X 2
        self.next_stage_input_cap = next_stage_input_cap
        self.frame_rate = frame_rate
        self.bus_length = num_cols * 1e-6 * 2

        self.write_counter = Countner(
            ADC_resolution=math.ceil(math.log2(self.num_cols*2)),
            input_clk_freq=self.input_clk_freq,
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.read_counter = Countner(
            ADC_resolution=math.ceil(math.log2(self.num_cols*2)),
            input_clk_freq=self.input_clk_freq,
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.bus = Wire(
            wire_type="semi_conservative",
            feature_size_nm=self.feature_size_nm,
            temperature=293
        )

        self.SRAM_cell = SRAM_cell(
            feature_size_nm=feature_size_nm,
            V_dd=self.V_dd
        )

        self.sense_amp  = Sense_Amp(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd
        )

        self.mux = MUX(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd,
            num_input=self.num_cols*2,
            next_stage_input_Cap=self.next_stage_input_cap
        )

        self.decoder = Decoder(
            feature_size_nm=self.feature_size_nm,
            V_dd=self.V_dd,
            num_input=self.num_cols*2,
            next_stage_input_Cap=0
        )

        self.read_power = self.compute_read_energy()*self.input_clk_freq
        self.write_power = self.compute_write_energy()*self.input_clk_freq
        self.bus_power = self.compute_bus_energy()*self.input_clk_freq
        self.counter_power = self.write_counter.total_power + self.read_counter.total_power
        self.mux_power = (self.mux.readDynamicEnergy + self.mux.writeDynamicEnergy)*self.input_clk_freq
        self.decoder_power = (self.decoder.readDynamicEnergy + self.decoder.writeDynamicEnergy)*self.input_clk_freq

        self.total_power=self.read_power+self.write_power+self.bus_power+self.counter_power+self.mux_power+self.decoder_power

    def compute_read_energy(self):
        Cap_bitline = (self.SRAM_cell.BL_cap + self.SRAM_cell.BL_bar_cap)*self.size 
        # print("cap bitline",Cap_bitline)
        Bitline_energy = (self.V_dd**2)*Cap_bitline

        Cap_wordline = self.SRAM_cell.WL_cap*self.size
        Wordline_energy = (self.V_dd**2)*Cap_wordline

        Cap_SA = (self.sense_amp.BL_bar_cap+self.sense_amp.BL_cap+self.sense_amp.output_cap+self.sense_amp.internal_cap)*self.num_cols
        SA_energy = (self.V_dd**2)*Cap_SA

        return Bitline_energy+Wordline_energy+SA_energy
    
    def compute_write_energy(self):
        Cap_bitline = (self.SRAM_cell.BL_cap + self.SRAM_cell.BL_bar_cap)*self.size 
        Bitline_energy = (self.V_dd**2)*Cap_bitline

        Cap_wordline = self.SRAM_cell.WL_cap*self.size
        Wordline_energy = (self.V_dd**2)*Cap_wordline

        Cap_Cell = self.SRAM_cell.CC_cap*self.size*0.5
        Cell_energy = (self.V_dd**2)*Cap_Cell

        return Bitline_energy+Wordline_energy+Cell_energy
    
    def compute_bus_energy(self):
        #calculate the delay and the energy of the input bus
        wire_capacitance = self.bus.cap_wire_per_m*self.bus_length
        energy = wire_capacitance*(self.V_dd ** 2)
        return energy