from cmath import atan
from os import stat
from platform import mac_ver
from numpy import cos, real, sin, tanh, arctan, arctan2, pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2, sawtooth, R, path_info_update, draw_crab, show_info, mat_reading, R3
import joystick


alpha = 1


def state(X):
    return alpha*X**2

# Drawing and window info
dt, w_size, w_shift = 0.1, 15, -7
fig, ax = plt.subplots(figsize=(8, 7))
state_info = []
T = []
x=-1
for t in np.arange(101, 200, dt):
    t_break = t
    T.append(t)
    state_info.append(x)
    x = x+dt*state(x)

ax.clear()
state_info = np.array(state_info)
T= np.array(T)
data = state_info
ax.plot(T, data, color='red')
ax.plot(T, 1/(1/(-1)-alpha*(T-101)), color='blue')
ax.set_xlabel('time (s)')
ax.set_ylabel('data')
plt.show()
