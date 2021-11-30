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

def weight_G_step_mapping(G_mapped, G_fit, n_states, G_min, G_max):
    G_step = (G_max - G_min) / n_states
    Rows , Cols = G_mapped.shape
    for row in range(Rows):
        for col in range(Cols):
            for state in range(n_states):
                if G_min + state * G_step <= G_mapped[row][col] < G_min + (state+1)*G_step:
                    G_mapped[row][col] = G_fit[state]
                    break
                # Maximum Conductance condition
                elif G_min + (n_states) * G_step <= G_mapped[row][col]:
                    G_mapped[row][col] = G_fit[n_states-1]
                    break
                # Minimum Conductance condition
                elif G_min + state * G_step > G_mapped[row][col]:
                    G_mapped[row][col] = G_min
                    break

    return G_mapped
