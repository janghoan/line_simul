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
# Weight
w = load_data_csv(data_path + 'WandB/10x10weight.csv')
b = load_data_csv(data_path + 'WandB/10x10bias.csv')
weight_mat = np.vstack((w,b))

#### 3. Memristor Device ####
# load CTM
CTM = CTM()
CTM.load()
# get fitted Conductance
G_fit = conduct(V = 10, G_max=CTM.G_max_device, G_min = CTM.G_min_device)

#### 4. Weight Mapping ####
G_mapped = weight_mapping(weight_mat = weight_mat, G_mat=G_fit)
print(1)















