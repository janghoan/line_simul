import numpy as np
from common.hw_utils import load_data_csv

class CTM():
    def __init__(self):
        self.CTM_device = None
        self.R_max_device = None
        self.R_min_device = None
        self.G_max_device = None
        self.G_min_device = None

    def load(self):
        self.CTM_device = load_data_csv('../Memristor_model/data/CTM_device/TiNbOx_CTM_potentiation.csv')
        self.CTM_device = self.CTM_device.T
        self.R_max_device = 1 / (self.CTM_device[-2][0])
        self.R_min_device = 1 / (self.CTM_device[-2][-1])
        self.G_max_device = 1 / self.R_min_device
        self.G_min_device = 1 / self.R_max_device

# voltage dependent term
def ap(V):
    aa=8.3326*(1e-6)*V**(5.68494)

    return aa*(1e-12)

# conductance change with Voltage (device characteristic)
def conduct(V,G_max,G_min):
    b=1
    G=G_min*(np.ones(35))
    for i in range(1,35):
        G[i]=G[i-1]+ap(V)*np.exp(-b*(G[i-1]-G_min)/(G_max-G_min))
    return G

# conductance change with Voltage in XBAR -> self rectifying characteristic
def conduct_self_rect(V,G_cell,G_max,G_min):
    # half voltage (5 V) condition
    if V>=5.1:
        b=1
        return ap(V)*np.exp(-b*(G_cell-G_min)/(G_max-G_min))

    elif V<5.1:
        return 0
