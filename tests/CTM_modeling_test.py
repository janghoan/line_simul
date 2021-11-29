import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Memristor_model.Charge_Trap_Memristor import *
from common.utils import load_data_csv

## 1. load device data ##
CTM  = load_data_csv('../Memristor_model/data/CTM_device/TiNbOx_CTM_potentiation.csv')
CTM = CTM.T

## 2. Max and Min Resistance and conductance of device (not fitted) ##
R_max_device = 1 / (CTM[-2][0])
R_min_device = 1 / (CTM[-2][-1])
G_max_device = 1/R_min_device
G_min_device = 1/R_max_device

## 3. number of states ##
n_pulse = 34
n_states = n_pulse

## 4. device fitting for 10 Voltage ##
# fitted conductance
G_fit = conduct(V = 10, G_max = G_max_device, G_min = G_min_device)

# Use max and min conductance as fitted value
G_max = max(G_fit)
G_min = min(G_fit)

plt.plot(G_fit, 'o')
plt.show()

## fitted graph for all voltage ##
G_fit_all = []
for V in [10,9,8,7,6,5]:
    G_fit_all.append(conduct(V = V, G_max = G_max_device, G_min = G_min_device))
G_fit_all = np.array(G_fit_all)
plt.plot(G_fit_all.T, 'o') # fitted graph
plt.plot(CTM.T, 'o', color = 'grey', alpha = 0.5) # device graph
plt.legend(['10','9','8','7','6','5'])
plt.show()



