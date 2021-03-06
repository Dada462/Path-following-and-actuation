from numpy import cos, sin, tanh, arctan, pi
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def rungeKutta2(x, u, h, f):
    x = x + h * (0.5 * f(x, u) + 0.5 * f(x + h * f(x, u), u))
    return x


def sawtooth(x):
    return (x + pi) % (2 * pi) - pi
    # or equivalently   2*arctan(tan(x/2))
    # return 2 * atan(math.tan(x/2))


class path_data:
    def __init__(self, X, psi, s, C_c):
        self.s = s
        self.psi = psi
        self.C_c = C_c
        self.X = X

    def __str__(self):
        return str(self.s) + ' ' + str(self.psi) + ' ' + str(self.C_c) + ' ' + str(self.X)


def mat_reading(path='PATH.mat'):  # Return the path for follow info: s,X,psi,C_c
    mat = scipy.io.loadmat(path)

    s = np.zeros(len(mat['Chemin']['s'][0, :]))
    for i in range(len(mat['Chemin']['s'][0, :])):
        s[i] = mat['Chemin']['s'][0, :][i][0][0]

    X = np.array([[], []])
    for i in range(len(mat['Chemin']['X'][0, :])):
        X = np.hstack((X, mat['Chemin']['X'][0, :][i]))
    X = X.T

    psi = np.zeros(len(mat['Chemin']['psi'][0, :]))
    for i in range(len(mat['Chemin']['psi'][0, :])):
        psi[i] = mat['Chemin']['psi'][0, :][i][0][0]

    C_c = np.zeros(len(mat['Chemin']['C_c'][0, :]))
    for i in range(len(mat['Chemin']['C_c'][0, :])):
        C_c[i] = mat['Chemin']['C_c'][0, :][i][0][0]

    path_to_follow = path_data(X, psi, s, C_c)
    return path_to_follow


path_to_follow = mat_reading()


def path_info_update(path_info_last, s):
    I, _ = next(((i, val) for (i, val) in enumerate(path_info_last.s) if val > s), (-1, -1))
    path_info = path_data(0, 0, 0, 0)
    if I < 0:
        print('End of the path')
        return None
    else:
        Delta_S_Lievre = path_info_last.s[I] - path_info_last.s[I - 1]
        ratio_S_Lievre = (s - path_info_last.s[I - 1]) / Delta_S_Lievre
        path_info.s = s
        path_info.psi = path_info_last.psi[I - 1] * (1 - ratio_S_Lievre) + path_info_last.psi[I] * ratio_S_Lievre
        path_info.C_c = path_info_last.C_c[I - 1] * (1 - ratio_S_Lievre) + path_info_last.C_c[I] * ratio_S_Lievre
        path_info.X = path_info_last.X[I - 1] * (1 - ratio_S_Lievre) + path_info_last.X[I] * ratio_S_Lievre
        return path_info


def state(X, u):
    _, _, vtx, vty, theta_m, dtheta_m = X.flatten()
    Vt = np.array([vtx, vty])

    A = 1 / m * np.array([[cos(alpha1), cos(alpha2), cos(alpha3), cos(alpha4)],
                          [sin(alpha1), sin(alpha2), sin(alpha3), sin(alpha4)],
                          [-d / J * cos(alpha1 + pi / 4), d / J * cos(alpha2 - pi / 4), -d / J * cos(alpha3 - 3 * pi / 4),
                           d / J * cos(alpha4 - 5 * pi / 4)]])
    b = np.vstack((-dtheta_m * (R(pi / 2) @ Vt).reshape((2, 1)), 0)).flatten()
    dVt_and_ddtheta_m = (A @ u + b).flatten()

    dVt = dVt_and_ddtheta_m[0:2]
    ddtheta_m = dVt_and_ddtheta_m[2]
    return np.hstack((R(theta_m) @ Vt, dVt, dtheta_m, ddtheta_m))


def R(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def draw_cube(P, theta, u):
    theta_deg = 180 * theta / pi - 90

    # The rectangular body
    body = plt.Rectangle(P - R(theta) @ np.array([a / 2, -b / 2]), a, b, angle=theta_deg, linewidth=1, edgecolor='g',
                         facecolor='none')

    # Plotting of the forces
    ax.quiver(*(P + R(theta) @ np.array([a / 2, b / 2])), *R(theta + alpha1)[:, 0] * u[0], color='red', scale=10)
    ax.quiver(*(P + R(theta) @ np.array([a / 2, -b / 2])), *R(theta + alpha2)[:, 0] * u[1], color='red', scale=10)
    ax.quiver(*(P + R(theta) @ np.array([-a / 2, -b / 2])), *R(theta + alpha3)[:, 0] * u[2], color='red', scale=10)
    ax.quiver(*(P + R(theta) @ np.array([-a / 2, b / 2])), *R(theta + alpha4)[:, 0] * u[3], color='red', scale=10)

    ax.add_patch(body)


def dyn_controller(x, path_to_follow):
    global end_of_path
    Vt=x[2:4]
    u,v = Vt
    theta_m,dtheta_m=x[4:6]

    X = x[0:2]
    M = np.array([[cos(theta_m), -sin(theta_m), 0], [sin(theta_m), cos(theta_m), 0], [0, 0, 1]])
    F = path_info_update(path_to_follow, s)
    if not F:
        end_of_path = 1
        return np.array([0, 0, 0]), 0
    theta_c = F.psi
    rotation = R(theta_c)
    s1, y1 = rotation.T @ (X - F.X)
    theta = theta_m - theta_c
    K_ds = 5
    ds = u * cos(theta) - sin(theta) * v + K_ds * s1

    dX_F = np.array([ds, 0])
    ds1, dy1 = (M[0:2, 0:2] @ rotation.T) @ Vt - dX_F
    dtheta_c = F.C_c * ds
    dtheta=dtheta_m-dtheta_c

    
    du=0
    Kdy1 = 1
    delta = -pi / 2 * tanh(Kdy1 * y1 * u)
    ddelta = -pi / 2 * Kdy1 * (1 - tanh(Kdy1 * y1 * u) ** 2) * (dy1*u+y1*du)
    # dddelta=-pi/2*(Kdy1*(ddu*y1+2*du*dy1 +u*ddy1)*(1-tanh(Kdy1*u*y1)**2) -2*Kdy1**2*(du*y1+u*dy1)**2*(1-tanh(Kdy1*u*y1)**2)*tanh(Kdy1*u*y1))
    
    # gamma = 0.5
    # k2=3
    # k3=2
    # zeta=ddelta-gamma*y1*u*(sin(theta) - sin(delta)) / sawtooth(theta - delta) - k2*sawtooth(theta-delta)
    # dzeta=dddelta - gamma*y1*u*((dtheta*cos(theta)-ddelta*cos(delta))*sawtooth(theta-delta)-(dtheta-ddelta)*(sin(theta)-sin(delta)))/(sawtooth(theta-delta)**2)-gamma*(dy1*u+du*y1)*(sin(theta) - sin(delta)) / sawtooth(theta - delta)-k2*(dtheta-ddelta)
    # eps=dtheta-zeta
    # deps=-1/gamma*sawtooth(theta-delta) - k3*eps
    # ddtheta=deps+dzeta

    # ddtheta=deps+dzeta+dtheta_c
    ddtheta_m=0

    k=10
    nu=3
    dnu=0
    Vtd=R(delta-theta)@np.array([nu,0])
    dVtd=R(delta-theta)@np.array([dnu,(ddelta-dtheta)*nu]).flatten()
    ddVt=(dVtd -y1*np.array([sin(theta),cos(theta)])-k*(Vt-Vtd)).reshape((2,1)).flatten()
    controller = np.array([ddVt[0],ddVt[1], ddtheta_m])
    A = 1 / m * np.array([[cos(alpha1), cos(alpha2), cos(alpha3), cos(alpha4)],
                          [sin(alpha1), sin(alpha2), sin(alpha3), sin(alpha4)],
                          [-d / J * cos(alpha1 + pi / 4), d / J * cos(alpha2 - pi / 4), -d / J * cos(alpha3 - 3 * pi / 4),
                           d / J * cos(alpha4 - 5 * pi / 4)]])
    b = np.vstack((-dtheta_m * (R(pi / 2) @ Vt).reshape((2, 1)), 0)).flatten()
    A_plus=np.linalg.pinv(A) #penrose inverse
    forces=A_plus@(controller-b).flatten()
    return forces, ds


#Properties of the model
m = 1
J = 1

d = 2**0.5

a, b = 1, 1

# Angle of the forces
alpha1 = pi/2
alpha2 = 0.8
alpha3 = pi
alpha4 = pi


# Drawing and window info
dt, w_size, w_shift = 0.01, 15, -7
fig, ax = plt.subplots(figsize=(9,8))

T, state_info = [], [] # T is the list temporal values, state_info[i] is the state x(t_i) where t_i=T[i]
t_break = 0  # it will be the time at which the simulation stops
vu, vv = 1, 0  # initial speed values along the u and v axis
x0 = np.array([-5, 10, vu, vv, 0, 0])  # (x,y,vu,vv,theta_m,dtheta_m)
u0 = np.array([0, 0, 0, 0])  # (f1,f2,f3,f4)
s0 = 0


path = [x0[0:2]]  # Red dots on the screen, path that the robot follows
x = x0
u = u0
s=s0

draw, end_of_path = 1, 0


for t in np.arange(0, 20, dt):
    u,ds=dyn_controller(x,path_to_follow)
    if end_of_path:
        print("End of simulation")
        break
    s = max(0, s)
    # Drawing and screen update
    if draw and (t / dt) % 5 == 0:
        ax.clear()
        ax.set_xlim(-w_size - w_shift, w_size - w_shift)
        ax.set_ylim(-w_size, w_size)
        draw_cube(x[0:2], x[4], u)
        ax.scatter(*path_info_update(path_to_follow, s).X, c='#34F44C')
        ax.plot(*path_to_follow.X.T, c='#3486F4')
        if (t / dt) % 30 == 0:  # Change the value to change the frequency of the red dots of the path
            path.append(x[0:2])
        for point in path:
            ax.scatter(*point, c='red', s=10)
        info = 'u1 : ' + str(round(u[0], 2)) + '\n' + 'u2 : ' + str(round(u[1], 2)) + '\n' + 'u3 : ' + str(
            round(u[2], 2))
        ax.text(w_size / 2, w_size + 0.5, info, style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
        ax.text(0, w_size + 1, 'time :' + str(round(t, 2)) + ' s', style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        plt.pause(10 ** -10)
    # Update of the state of the simulation
    T.append(t)
    state_info.append(x)
    x = rungeKutta2(x, u, dt, state)
    s = s + ds * dt
    t_break = t

# Plotting results
ax.clear()
data1=np.array(state_info)[:,2]
data2=np.array(state_info)[:,3]
data3=np.array(state_info)[:,5]
ax.plot(T, data1,color='red')
ax.plot(T, data2,color='blue')
ax.plot(T, data3,color='green')
ax.set_xlabel('time (s)')
ax.set_ylabel('data')
plt.show()
