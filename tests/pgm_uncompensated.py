import os, sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import tensorflow as tf
from scipy import linalg
from common.hw_utils import *
from Memristor_model.Charge_Trap_Memristor import *
from XBAR_model.XBAR_calculation import XBAR_ARRAY

data_path = '../MNIST_SW/data/'

#### 1. load data ####
# load train_data

train_data = load_data_csv(data_path + 'train_and_test_data/1010_v2.csv')
t_train, x_train = tf.one_hot(train_data[:, 0], depth = 10), train_data[:, 1:]/255

# load test_data
test_data = load_data_csv(data_path + 'train_and_test_data/1010t_v2.csv')
t_test, x_test = tf.one_hot(test_data[:, 0], depth = 10), test_data[:, 1:]/255

#### 2. load Weight and Bias ####
w = load_data_csv(data_path + 'WandB/10x10weight.csv')
b = load_data_csv(data_path + 'WandB/10x10bias.csv')
weight_mat = np.vstack((w, b))

#### 3. Memristor Device ####
# load CTM
CTM = CTM()
CTM.load()
# get fitted Conductance
G_fit = conduct(V = 10, G_max=CTM.G_max_device, G_min = CTM.G_min_device)
G_max = np.max(G_fit)
G_min = np.min(G_fit)

#### 4. Weight Mapping ####
# mapping weight
G_mapped_weight = weight_mapping(weight_mat = w, G_mat=G_fit)
G_mapped_weight_ideal = G_mapped_weight.copy()
# mapping bias
G_mapped_bias = weight_mapping(weight_mat = b, G_mat=G_fit)
G_mapped_bias_ideal = G_mapped_bias.copy()


#### 5. Mapping with Device ####
n_pulse = 34
n_states = n_pulse
# Mapping weight with G_step
G_mapped_weight = weight_G_step_mapping(G_mapped = G_mapped_weight, G_fit = G_fit, n_states = n_states, G_min = G_min, G_max = G_max)
# Mapping bias with G_step
G_mapped_bias = weight_G_step_mapping(G_mapped = G_mapped_bias, G_fit = G_fit, n_states = n_states, G_min = G_min, G_max = G_max)

#### 6. Get G Index for Programming ####
# weight
G_index = get_G_index(G_mapped=G_mapped_weight, G_fit = G_fit)
# bias
G_bias_index = get_G_index(G_mapped=G_mapped_bias, G_fit = G_fit)

#### 7. Voltages and Line Resistance ####
'''
    WL1 = Left Side of the Word line
    WL2 = Right Side of the Word line
    BL1 = Top of the Bit line
    BL2 = Bottom of the Bit line
'''
V_APP_WL1 = 10 # WL input voltage
V_APP_BL2 = 0

# initial Resistance -> All HRS cells
R_device_init = (1/G_min) * np.ones((weight_mat.shape))
XBAR = XBAR_ARRAY(Device_R=R_device_init.copy())
XBAR.R_min, XBAR.R_max  = (1/G_max), (1/G_min)

#### 8. Programming with pulse number ####

#Fixme
# ########################################################################
# V_device_lst= []
# for coll in range(10,110,10):
#     G_index = np.random.randint(0,34, (128,coll))
#     G_bias_index = np.random.randint(0, 34, (1, coll))
#     R_device = (1/G_max) *np.ones((129,coll))
# ########################################################################

### G_ideal Programming ###
r_factor = 1e-8
XBAR.V_WL, XBAR.R_S_WL  = r_factor * (1/G_min), r_factor * (1/G_min)
XBAR.programming( V_APP_WL = V_APP_WL1, G_index = G_index, G_bias_index = G_bias_index )
G_ideal = 1 / XBAR.Device_R

### G Programming ###
r_factor = 1e-4 # r_factor = R_Line / R_LRS
XBAR.Device_R = R_device_init.copy()
XBAR.V_WL, XBAR.R_S_WL  = r_factor * (1/G_min), r_factor * (1/G_min)
XBAR.programming( V_APP_WL = V_APP_WL1, G_index = G_index, G_bias_index = G_bias_index )
G_1e_4 = 1 / XBAR.Device_R

# GGG=1/(XBAR.Device_R[:weight_mat.shape[0]-1])
# plt.scatter(w,GGG,s=1.5)
# plt.ylim(0,G_max+0.5*G_max)
# plt.xlabel('weight')
# plt.ylabel('conductance')
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data',0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',0))
# plt.show()

# plt.plot(G_ideal, G_ideal, '-o',color = 'r')
# plt.plot(G_ideal, G_1e_4, 'o',color = 'b')
# plt.show()

plt.plot(G_ideal * 1e+12,G_ideal* 1e+12,'-o',ms=2.5,color='r',label='G ideal')
plt.plot(G_ideal* 1e+12,G_1e_4* 1e+12,'o',ms=2.5,color='b',label='G read')
#plt.plot(G_ideal,G_read_comp_node_10M,'o',ms=2.5,color='m',label='G read')
plt.ylabel('G pgm',fontsize=23)
plt.xlabel('G ideal',fontsize=23)
plt.title('G ideal vs G pgm',fontsize=25)
plt.show()




# V_device_lst.append(XBAR.V_device)
# sns.heatmap(R_device)
# plt.show()
# V_d = np.zeros((10,129,100))
# for i in range(len(V_device_lst)):
#     for j in range(len(V_device_lst[i])):
#         for k in range(len(V_device_lst[i][j])):
#             V_d[i][j][k] = V_device_lst[i][j][k]
#
# # i = 0
# # for cols in range(10,110,10):
# #     plt.plot(V_d[i][-1][:cols],'o')
# #     i+=1
# # plt.show()
#
# cnt = 10
# for j in range(10):
#     for i in range(len(V_d[j][-1][:cnt])):
#         plt.plot(V_d[j][i][:cnt])
#     cnt+=10
#     plt.show()
print(1)
















