import pandas as pd
import numpy as np

def load_data_csv(path):
    data = pd.read_csv(path)
    data = np.array(data)
    return data

def weight_mapping(weight_mat, G_mat):
    w_min, w_max = np.min(weight_mat), np.max(weight_mat)
    G_min, G_max = np.min(G_mat), np.max(G_mat)
    slope = (G_max - G_min) / (w_max - w_min)
    y_intercept = G_max - slope * w_max
    G_mapped = slope * weight_mat + y_intercept

    return G_mapped