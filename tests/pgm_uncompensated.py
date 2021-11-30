import os, sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import tensorflow as tf
from scipy import linalg
from common.utils import *
from Memristor_model.Charge_Trap_Memristor import *

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




















