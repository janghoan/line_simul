import sys
import numpy as np
import matplotlib.pyplot as plt

import random
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from Memristor_model.Charge_Trap_Memristor import conduct_self_rect
import csv

class XBAR_ARRAY:
    def __init__(self, Device_R =None):
        # self.Row = Row
        # self.Col = Col
        self.Row, self.Col = Device_R.shape
        self.Device_R = Device_R
        self.V_WL = None
        self.R_S_WL = None
        self.R_max = None
        self.R_min = None
        self.V_device = None

    def programming(self, V_APP_WL, G_index, G_bias_index ):
        self.V_device = np.zeros((self.Row, self.Col))
        for row in range(self.Row):
            for col in range(self.Col):
                # get n_pulse
                if row != self.Row - 1:
                    n_pulse = int(G_index[row][col])
                else:  # last low
                    n_pulse = int(G_bias_index[-1,][col])

                for pulse in range(0, n_pulse + 1):
                    self.V_device[row][col] = V_APP_WL * self.Device_R[row][col] / ((self.Row-row)*self.V_WL+self.Device_R[row][col]+(col+1)*self.R_S_WL)

                    if pulse == 0:
                        if self.V_device[row][col] < -5:
                            self.Device_R[row][col] = self.Device_R[row][col] * 10 ** 4
                        else:
                            self.Device_R[row][col] = self.Device_R[row][col]

                    elif pulse > 0:
                        if self.V_device[row][col] < -5:
                            self.Device_R[row][col] = self.Device_R[row][col] * 10 ** 4
                        else:
                            self.Device_R[row][col] = 1 / (1/self.Device_R[row][col] + conduct_self_rect(V = self.V_device[row][col], G_cell =  1/self.Device_R[row][col], G_max = 1/self.R_min, G_min = 1/self.R_max ))

                    # if self.V_device[row][col] < -5:
                    #     self.Device_R[row][col] = self.Device_R[row][col] * 10 ** 4
                    # else:
                    #     if pulse == 0:
                    #         self.Device_R[row][col] = self.Device_R[row][col]
                    #     elif pulse > 0:
                    #         self.Device_R[row][col] = 1 / (1/self.Device_R[row][col] + conduct_self_rect(V = self.V_device[row][col], G_cell =  1/self.Device_R[row][col], G_max = 1/self.R_min, G_min = 1/self.R_max ))




    def Output_current(self):
        print("Start Calculation ... ")
        m = self.Row
        n = self.Col
        R_d =self.Device_R
        V_APP_WL1 =self.V_WL * np.ones(m)
        R_S_WL1 =self.R_S_WL

        mn = m*n
        R_WL = R_S_WL1
        R_S_BL2 = R_S_WL1
        R_BL = R_S_WL1
        V_APP_BL2 = 0

        A = np.zeros((mn, mn))  # 10000*10000 영행렬
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if j == 1:
                    A[n * (i - 1) + j - 1][n * (i - 1) + j - 1] = (1 / R_S_WL1) + (1 / R_d[i - 1][j - 1]) + 1 / R_WL
                    A[n * (i - 1) + j - 1][n * (i - 1) + j] = -1 / R_WL
                elif 1 < j < n:
                    A[n * (i - 1) + j - 1][n * (i - 1) + j - 2] = -1 / R_WL
                    A[n * (i - 1) + j - 1][n * (i - 1) + j - 1] = (1 / R_d[i - 1][j - 1]) + 2 / R_WL  # -3
                    A[n * (i - 1) + j - 1][n * (i - 1) + j] = -1 / R_WL
                elif j == n:
                    A[n * (i - 1) + j - 1][n * (i - 1) + j - 2] = -1 / R_WL
                    A[n * (i - 1) + j - 1][n * (i - 1) + j - 1] = 0 + (1 / R_d[i - 1][j - 1]) + 1 / R_WL

        B = np.zeros((mn, mn))
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                B[n * (i - 1) + j - 1][n * (i - 1) + j - 1] = -1 / R_d[i - 1][j - 1]

        AB = np.hstack((A,B))
        del A
        del B

        # C
        C = np.zeros((mn, mn))
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                C[m * (j - 1) + i - 1][n * (i - 1) + j - 1] = 1 / R_d[i - 1][j - 1]
        # D
        D = np.zeros((mn, mn))
        for j in range(1, n + 1):
            for i in range(1, m + 1):

                if i == 1:
                    D[m * (j - 1) + i - 1][j - 1] = (-1 / R_BL) + (-1 / R_d[i - 1][j - 1])
                    D[m * (j - 1) + i - 1][n * i + j - 1] = 1 / R_BL
                elif 1 < i < m:
                    D[m * (j - 1) + i - 1][n * (i - 2) + j - 1] = 1 / R_BL
                    D[m * (j - 1) + i - 1][n * (i - 1) + j - 1] = (-1 / R_BL) + (-1 / R_d[i - 1][j - 1]) + (-1 / R_BL)
                    D[m * (j - 1) + i - 1][n * (i) + (j - 1)] = 1 / R_BL
                elif i == m:
                    D[m * (j - 1) + i - 1][n * (i - 2) + j - 1] = 0 + 1 / R_BL
                    D[m * (j - 1) + i - 1][n * (i - 1) + j - 1] = (-1 / R_S_BL2) + (-1 / R_d[i - 1][j - 1]) + (-1 / R_BL)

        # CD만들기
        CD = np.hstack((C, D))
        del C
        del D

        # ABCD만들기
        ABCD = np.vstack((AB, CD))
        del AB
        del CD

        # E : input voltage
        E = np.zeros((2 * mn, 1))  # 추가 필요!!!

        for i in range(1, m + 1):
            E[(i - 1) * n][0] = V_APP_WL1[i - 1] / R_S_WL1
            E[n - 1 + (i - 1) * n][0] = 0

        for j in range(1, n + 1):
            E[(n - 1) + (m - 1) * n + 1 + (j - 1) * m][0] = 0
            E[(n - 1) + (m - 1) * n + j * m][0] = -V_APP_BL2 / R_S_BL2

        V = np.linalg.solve(ABCD, E)

        V_array = V
        V_WL_NODE_array = V_array[0:mn]
        V_BL_NODE_array = V_array[mn:2 * mn]
        V_devices_array = V_WL_NODE_array - V_BL_NODE_array

        Vdevices = V_devices_array.reshape(m, n)

        V_where = np.ones((Vdevices.shape[0], Vdevices.shape[1]))
        R_where = np.ones((Vdevices.shape[0], Vdevices.shape[1]))

        for i in range(m):
            for j in range(n):
                if Vdevices[i][j] < 0:
                    V_where[i][j] = 0
                    R_d[i][j] = 1e+16
                    R_where[i][j] = 0

        i = 1

        # apply selector characteristics : self-rectifying current
        while np.sum(V_where - R_where) != 0:

            A = np.zeros((mn, mn))  # 10000*10000 영행렬
            for i in range(1, m + 1):
                for j in range(1, n + 1):

                    if j == 1:
                        A[n * (i - 1) + j - 1][n * (i - 1) + j - 1] = (1 / R_S_WL1) + (1 / R_d[i - 1][j - 1]) + 1 / R_WL
                        A[n * (i - 1) + j - 1][n * (i - 1) + j] = -1 / R_WL
                    elif 1 < j < n:
                        A[n * (i - 1) + j - 1][n * (i - 1) + j - 2] = -1 / R_WL
                        A[n * (i - 1) + j - 1][n * (i - 1) + j - 1] = (1 / R_d[i - 1][j - 1]) + 2 / R_WL  # -3
                        A[n * (i - 1) + j - 1][n * (i - 1) + j] = -1 / R_WL
                    elif j == n:
                        A[n * (i - 1) + j - 1][n * (i - 1) + j - 2] = -1 / R_WL
                        A[n * (i - 1) + j - 1][n * (i - 1) + j - 1] = 0 + (1 / R_d[i - 1][j - 1]) + 1 / R_WL

            # B
            B = np.zeros((mn, mn))
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    B[n * (i - 1) + j - 1][n * (i - 1) + j - 1] = -1 / R_d[i - 1][j - 1]

            # AB만들기
            AB = np.hstack((A, B))
            del A
            del B

            # C
            C = np.zeros((mn, mn))
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    C[m * (j - 1) + i - 1][n * (i - 1) + j - 1] = 1 / R_d[i - 1][j - 1]
            # D
            D = np.zeros((mn, mn))
            for j in range(1, n + 1):
                for i in range(1, m + 1):

                    if i == 1:
                        D[m * (j - 1) + i - 1][j - 1] = (-1 / R_BL) + (-1 / R_d[i - 1][j - 1])
                        D[m * (j - 1) + i - 1][n * i + j - 1] = 1 / R_BL
                    elif 1 < i < m:
                        D[m * (j - 1) + i - 1][n * (i - 2) + j - 1] = 1 / R_BL
                        D[m * (j - 1) + i - 1][n * (i - 1) + j - 1] = (-1 / R_BL) + (-1 / R_d[i - 1][j - 1]) + (-1 / R_BL)
                        D[m * (j - 1) + i - 1][n * (i) + (j - 1)] = 1 / R_BL
                    elif i == m:
                        D[m * (j - 1) + i - 1][n * (i - 2) + j - 1] = 0 + 1 / R_BL
                        D[m * (j - 1) + i - 1][n * (i - 1) + j - 1] = (-1 / R_S_BL2) + (-1 / R_d[i - 1][j - 1]) + (
                                    -1 / R_BL)

            # CD만들기
            CD = np.hstack((C, D))
            del C
            del D

            # ABCD만들기
            ABCD = np.vstack((AB, CD))
            del AB
            del CD

            E = np.zeros((2 * mn, 1))  # 추가 필요!!!

            for i in range(1, m + 1):
                E[(i - 1) * n][0] = V_APP_WL1[i - 1] / R_S_WL1
                E[n - 1 + (i - 1) * n][0] = 0

            for j in range(1, n + 1):
                E[(n - 1) + (m - 1) * n + 1 + (j - 1) * m][0] = 0
                E[(n - 1) + (m - 1) * n + j * m][0] = -V_APP_BL2 / R_S_BL2

            # V(i,j) 계산하기
            V = np.linalg.solve(ABCD, E)


            # eval() : V_M 텐서를 어레이로 변환! sess.close()하지 않은 상태에서

            V_array = V
            V_WL_NODE_array = V_array[0:mn]
            V_BL_NODE_array = V_array[mn:2 * mn]

            V_devices_array = V_WL_NODE_array - V_BL_NODE_array

            Vdevices = V_devices_array.reshape((m, n))

            # sns.heatmap(Vdevices)

            V_where = np.ones((Vdevices.shape[0], Vdevices.shape[1]))
            R_where = np.ones((Vdevices.shape[0], Vdevices.shape[1]))

            for i in range(np.shape(Vdevices)[0]):
                for j in range(np.shape(Vdevices)[1]):
                    if Vdevices[i][j] < 0:
                        V_where[i][j] = 0
                        R_d[i][j] = 1e+16 # reverse resistance (Charge Trap Memristor)
                        R_where[i][j] = 0
            i += 1

        Iout = (V_BL_NODE_array.reshape((m, n))[m - 1] / R_BL)
        print("Calculation Done !")

        return Iout





