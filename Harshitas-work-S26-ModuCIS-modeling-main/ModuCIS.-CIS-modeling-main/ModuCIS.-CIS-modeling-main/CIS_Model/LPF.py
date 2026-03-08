# Low-Pass Filter (LPF) class
# Models a low-pass filter circuit used in PLL for loop filtering
# 
# Inputs:
#   Filter Parameters:
#     - resistance (filter resistance in Ω)
#   Capacitance Parameters:
#     - load_cap (load capacitance in F)
# Outputs: (no outputs - filter parameters only)
# A basic low-pass filter that is made up of resistors and capacitors.
class LPF:
    def __init__(self, resistance, load_cap):
        self.resistance = resistance
        self.load_cap = load_cap