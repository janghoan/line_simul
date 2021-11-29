from XBAR_model.XBAR_calculation import XBAR_ARRAY
import numpy as np
import matplotlib.pyplot as plt
# model 1: 128 x 64 Crossbar array
Row = 32
Col = 32
Device_R = int(200) * np.ones((Row, Col))
V_WL = 1
R_S_WL = 1
# XBAR = XBAR_ARRAY(Row, Col, Device_R, V_WL, R_S_WL)
XBAR = XBAR_ARRAY(Row = Row , Col = Col, Device_R = Device_R, V_WL= V_WL, R_S_WL = R_S_WL)


Output_Current = XBAR.Output_current()

plt.plot(Output_Current, 'o-', ms = 2)
plt.show()
print(Output_Current)
print(1)
