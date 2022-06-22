from cmath import atan
from numpy import arctan2, cos, sin, tanh, arctan, pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2, sawtooth, mat_reading, R, path_info_update, draw_crab, show_info, key_press, rect


def state(X, controller):
    _, _, theta_m, s = X.flatten()
    u = controller[0:-1]
    ds = controller[-1]
    *Vt, dtheta_m = A@D@u.flatten()
    return np.hstack((R(theta_m)@Vt, dtheta_m, ds))


def controller(x, U):
    global end_of_path,dbeta
    X = x[0:2]
    u_last = U[-1]
    Vt = (A@D@u_last[0:-1])[0:2]
    s = x[3]
    theta_m = x[2]

    F = path_info_update(path_to_follow, s)
    if F == None:
        end_of_path = 1
        return np.array([0, 0, 0, 0])
    theta_c = F.psi
    theta = theta_m-theta_c

    s1, y1 = R(theta_c).T@(X-F.X)

    ks = 3
    ds = np.dot(R(theta)[0, :], Vt) + ks*s1
    dtheta_c = F.C_c*ds
    ds1, dy1 = R(theta)@Vt-ds*np.array([1-F.C_c*y1, F.C_c*s1])

    Kdy1 = 1
    nu=3
    dnu=0
    delta = -pi/2*tanh(Kdy1*y1*nu)
    ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*nu)**2)*(dy1*nu+dnu*dy1)
    

    gamma = 2
    k2 = 2
    beta = arctan2(Vt[1], Vt[0])
    a = Vt[0]**2+Vt[1]**2
    u_last_last = U[-2]
    Vt_last = (A@D@u_last_last[0:-1])[0:2]
    dVt=(Vt-Vt_last)/dt

    if a != 0:
        dbeta = (dVt[1]*Vt[0]-dVt[0]*Vt[1])/a
    else:
        dbeta = 0

   
    psi = theta+beta

    def f(x, y):
        if x != y:
            return (sin(x) - sin(y)) / sawtooth(x - y)
        else:
            return 1
    dpsi = ddelta-gamma*y1*nu*f(psi, delta) - k2*sawtooth(psi-delta)
    dtheta_m = dpsi+dtheta_c-dbeta

    Vtd = R(delta-psi)@np.array([3, 0])
    w1, w2, w3 = np.linalg.pinv(A@D)@np.array([*Vtd, dtheta_m])

    return np.array([w1, w2, w3, ds])


# Geometric parameters of the robot
alpha2 = 2*pi/3
alpha3 = -2*pi/3
d1 = d2 = d3 = 1
# d1=0

L = 1
r = 1
D = np.array([[d1, 0, 0], [0, d2, 0], [0, 0, d3]])
B = 1/r*np.array([[0, 1, L], [-sin(alpha2), cos(alpha2), L],
                 [-sin(alpha3), cos(alpha3), L]])
A = np.linalg.inv(B)


# Drawing and window info
dt, w_size, w_shift = 0.01, 15, -7
fig, ax = plt.subplots(figsize=(8, 7))

# Lists to stock the information. For viewing resulsts after the simulation
# T is the list temporal values, state_info[i] is the state x(t_i) where t_i=T[i]
T, state_info = [], []
t_break = 0  # it will be the time at which the simulation stops

# Initial conditions
px0, py0 = -5, 10
theta0 = 0
s0 = 0
omega1, omega2, omega3 = 0, 0, 0


path_to_follow = mat_reading()  # The path the robot has to follow
x0 = np.array([px0, py0, theta0, s0])  # (x,y,vu,vv,theta_m,dtheta_m)
u0 = np.array([omega1, omega2, omega3, 0])  # (f1,f2,f3)
path = [x0[0:2]]  # Red dots on the screen, path that the robot follows

key = np.array([0, 0, 0])  # To control the robot using the keyboard
x = x0
u = u0

U = [u0]*5
data = []
draw, end_of_path = 1, 0

Vt_last = np.array([0, 0])
dbeta=0
for t in np.arange(0, 20, dt):
    if end_of_path:
        print("End of simulation")
        break
    # Drawing and screen update
    X = x[0:2]
    Vt = (A@D@u[0:-1])[0:2]
    theta_m = x[2]
    s = x[3]
    # beta=arctan2(Vt[1],Vt[0])
    # dVt=(Vt-Vt_last)/dt
    # if np.linalg.norm(Vt.reshape((2,1)))!=0:
    #     a=Vt[0]**2+Vt[1]**2
    #     dbeta=(dVt[1]*Vt[0]-dVt[0]*Vt[1])/a
    # else:
    #     dbeta=0
    # data.append([beta,dbeta])

    if draw and (t/dt) % 5 == 0:
        ax.clear()
        ax.set_xlim(-w_size-w_shift, w_size-w_shift)
        ax.set_ylim(-w_size, w_size)
        show_info(ax, path_to_follow, X, Vt, theta_m, -u, [0, alpha2, alpha3, L, r], [
                  s, t], path, [w_size, w_shift], forces=False, speed=True)
        draw_crab(X, theta_m, ax, 0.5, 0.25)
        if (t/dt) % 30 == 0:  # Change the value to change the frequency of the red dots of the path
            path.append(X)
        ax.text(w_size/2, -w_size+1, 'dbeta: ' +str(dbeta), style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
        
        plt.pause(10**-10)
    # Update of the state of the simulation
    t_break = t
    T.append(t)
    state_info.append(x)
    x = rungeKutta2(x, u, dt, state)
    if t/dt % 2 == 0:
        u = controller(x, U)
    U.append(u)
    # Vt_last=Vt
    # Vt = (A@D@u[0:-1])[0:2]
    x[3] = max(0, x[3])
    x[2] = sawtooth(x[2])

# Plotting results
ax.clear()
beta = np.array(data)[:, 0]
dbeta = np.array(data)[:, 1]


def f(X):
    s = 0
    for x in X:
        s += x
    return s


beta1 = [f(dbeta[0:i])*dt-beta[0] for i in range(len(dbeta))]
ax.plot(T, beta, color='red')
ax.plot(T, beta1, color='blue')
ax.set_xlabel('time (s)')
ax.set_ylabel('data')
plt.show()
