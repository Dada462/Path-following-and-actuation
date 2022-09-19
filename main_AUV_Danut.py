from cmath import atan
from platform import mac_ver
from numpy import cos, real, sin, tanh, arctan, arctan2, pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2, sawtooth, R, path_info_update, draw_crab, show_info,mat_reading,mat_reading1,R3
import joystick



def state(X, controller):
    _, _, vtx, vty, s, theta_m, dtheta_m = X.flatten()
    u = controller[0:-1]
    Vt = np.array([vtx, vty])
    Fu,Fv,Gamma=A@u
    ds = controller[-1]
    d_u = -Xuu*vtx**2-Xvv*vty**2
    d_v = -Yv*vtx*vty-Yvabsv*vty*abs(vty)
    d_r = -Nv*vtx*vty-Nvabsv*vty*abs(vty)-Nr*vtx*dtheta_m
    du = (Fu-d_u)/mu
    dv = (Fv-d_v-mur*vtx*dtheta_m)/mv
    ddtheta_m = (Gamma-d_r)/mr
    dVt = np.array([du, dv])
    return np.hstack((R(theta_m)@Vt, dVt, ds, dtheta_m, ddtheta_m))


def controller_Col(x):
    #For the case when two motors are colinear
    global end_of_path
    X = x[0:2]
    Vt = x[2:4]
    s = x[4]
    theta_m, dtheta_m = x[5:7]
    u, v = Vt
    nu = (u**2+v**2)**0.5

    d_v = -Yv*u*v-Yvabsv*v*abs(v)
    dVt=np.array([2*(1-u),-(d_v+mur*u*dtheta_m)/mv])
    du,dv=dVt
    dd_v = -Yv*u*dv-2*Yvabsv*abs(v)
    d_u=-Xuu*u**2-Xvv*v**2
    d_r=-Nv*u*v-Nvabsv*v*abs(v)-Nr*u*dtheta_m
    beta_mes = arctan2(v, u)
    dbeta_mes = (u*dv-du*v)/(nu**2)

    dnu =cos(beta_mes)*du+sin(beta_mes)*dv

    F = path_info_update(path_to_follow, s)
    theta_c = F.psi
    s1, y1 = R(theta_c).T@(X-F.X)
    theta = sawtooth(theta_m-theta_c)
    psi = sawtooth(theta+beta_mes)
    ks = 1
    ds = cos(psi)*nu + ks*s1
    dtheta_c = F.C_c*ds
    ds1, dy1 = R(theta)@Vt-ds*np.array([1-F.C_c*y1, F.C_c*s1])
    dtheta = dtheta_m-dtheta_c
    dpsi = dtheta+dbeta_mes
    dds = -dpsi*sin(psi)*nu+cos(psi)*dnu+ks*ds1
    ddtheta_c = F.dC_c*(ds**2)+F.C_c*dds
    dds1,ddy1=R(psi)@np.array([dnu,dpsi*nu])-dds*np.array([1-F.C_c*y1,F.C_c*s1])-ds*np.array([-F.dC_c*y1*ds-F.C_c*dy1,F.dC_c*s1*ds+F.C_c*ds1])


    Kdy1 = 1
    psi_a=pi/2
    delta = -psi_a*tanh(Kdy1*y1)
    ddelta = -psi_a*Kdy1*(1-tanh(Kdy1*y1)**2)*dy1
    dddelta = -psi_a*Kdy1*((1-tanh(Kdy1*y1)**2)*ddy1-2 *
                          Kdy1*dy1**2*tanh(Kdy1*y1)*(1-tanh(Kdy1*y1)**2))

    k1 = 1
    k3 = 1
    k5 = 1
    zeta = dddelta+ddtheta_c+(k1+k3)*(ddelta-dpsi)+(k5+k3*k1)*sawtooth(delta-psi)
    ddbeta = (-u/(nu**2*mv)*(mur*u*zeta+dd_v+mur*du*dtheta_m)-2*dnu*(u*dv-du*v)/(nu**3)-du)/(1-(cos(beta_mes)**2)*mur/mv)

    ddtheta_m = dddelta-ddbeta+ddtheta_c + (k1+k3)*(ddelta-dpsi)+(k5+k3*k1)*sawtooth(delta-psi)

    F=A_plus@np.array([(d_u+2*(1-u)*mu),0,mr*ddtheta_m+d_r])
    return np.array([*F,ds])


def controller_Non_Col(x):
    #For the case when at least two motors are not colinear
    global end_of_path
    X = x[0:2]
    Vt = x[2:4]
    s = x[4]
    theta_m, dtheta_m = x[5:7]
    u, v = Vt
    nu = (u**2+v**2)**0.5
    beta=arctan2(v,u)

    d_v = -Yv*u*v-Yvabsv*v*abs(v)
    d_u=-Xuu*u**2-Xvv*v**2
    d_r=-Nv*u*v-Nvabsv*v*abs(v)-Nr*u*dtheta_m
    
    F = path_info_update(path_to_follow, s)
    theta_c = F.psi
    s1, y1 = R(theta_c).T@(X-F.X)
    theta = sawtooth(theta_m-theta_c)
    psi = sawtooth(theta+beta)
    ks = 1
    ds = cos(psi)*nu + ks*s1
    dtheta_c = F.C_c*ds
    ds1, dy1 = R(theta)@Vt-ds*np.array([1-F.C_c*y1, F.C_c*s1])

    psi_a=pi/2
    Kdy1=1
    delta = -psi_a*tanh(Kdy1*y1)
    ddelta = -psi_a*Kdy1*(1-tanh(Kdy1*y1)**2)*dy1

    dnu=2*(1-nu)
    dpsi=ddelta+2*sawtooth(delta-psi)
    dbeta=dpsi-dtheta_m+dtheta_c
    theta_m_d=-pi/2
    dtheta_m_d=0
    ddtheta_m_d=0
    ddtheta_m=ddtheta_m_d+2*(dtheta_m_d-dtheta_m)+1*(theta_m_d-theta_m)
    d=np.array([d_u/mu,(d_v+mur*u*dtheta_m)/mv,d_r/mr])
    M=np.array([[mu,0,0],[0,mv,0],[0,0,mr]])
    F=np.linalg.pinv(zeta(A)@np.linalg.inv(M)@A)@zeta(A)@(R3(beta)@np.array([dnu,nu*dbeta,ddtheta_m])+d)
    return np.array([*F,ds])


def zeta(A):
    if np.linalg.matrix_rank(A)<=1:
        return np.zeros((3,3))
    elif np.linalg.matrix_rank(A)==2:
        return np.array([[1,0,0],[0,1,0]])
    elif np.linalg.matrix_rank(A)>2:
        return np.eye(3)
# def zeta_1(x):
#     u, v = x[2:4]
#     nu = (u**2+v**2)**0.5
#     beta=arctan2(v,u)
#     M=np.array([[mu,0,0],[0,mv,0],[0,0,mr]])
#     if 13:
#         return np.array([[1,0,0],[0,1/nu,0],[0,0,1]])@R3(beta).T@np.linalg.inv(M)
#     else:
#         return np.linalg.inv(M)
# def zeta_2(nu,delta,ddelta,psi,dtheta_c,theta_m,dtheta_m):
#     dnu=2*(1-nu)
#     dpsi=ddelta+2*sawtooth(delta-psi)
#     dbeta=dpsi-dtheta_m+dtheta_c
#     theta_m_d=-pi/2
#     dtheta_m_d=0
#     ddtheta_m_d=0
#     ddtheta_m_orientation=ddtheta_m_d+2*(dtheta_m_d-dtheta_m)+1*(theta_m_d-theta_m)
#     if 13:
#         return np.array([dnu,dbeta,ddtheta_m_orientation])
#     elif 24:
#         u_rho_d=1
#         du_rho=0
#         k_rho=1
#         du_rho=du_rho+k_rho*(u_rho_d-u_rho)
#         zeta = dddelta+ddtheta_c+(k1+k3)*(ddelta-dpsi)+(k5+k3*k1)*sawtooth(delta-psi)
#         ddbeta = (-u/(nu**2*mv)*(mur*u*zeta+dd_v+mur*du*dtheta_m)-2*dnu*(u*dv-du*v)/(nu**3)-du)/(1-(cos(beta_mes)**2)*mur/mv)
#         ddtheta_m_path_following = dddelta-ddbeta+ddtheta_c + (k1+k3)*(ddelta-dpsi)+(k5+k3*k1)*sawtooth(delta-psi)
#         return np.array([du_rho,ddtheta_m_path_following])
        
# Geometric parameters of the robot
Xuu = -35
Xvv = -128
Yv = -346
Yvabsv = -667
Nv = -686
Nvabsv = 443
Nr = -1427
Yr = 435
Ydv = -1715
Xdu = -142
Ndr = -1350
Iz = 2000

m = 2234
mur = m-Yr
mv = m-Ydv
mu = m-Xdu
mr = Iz-Ndr


alpha2=2*pi/3
alpha3=-2.5
d1=d2=d3=1
d2=0
L=1
r=0.25 #Size of the robot on the drawings, purely for drawing purposes
D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
A=np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-L,-L,-L]])@D
A_plus=np.linalg.pinv(A) #penrose inverse

# Drawing and window info
dt, w_size, w_shift = 0.01, 20, 0
fig, ax = plt.subplots(figsize=(8, 7))

# Lists to stock the information. For viewing resulsts after the simulation
# T is the list temporal values, state_info[i] is the state x(t_i) where t_i=T[i]
T, state_info = [], []
t_break = 0  # it will be the time at which the simulation stops

# Initial conditions
px0, py0 = -5, 10 #Initial position
u0, v0 = 1,0 #Initial speed
theta0, dtheta0 = 0, 0 # Initial orientation and angular velocity
s0 = 0



def spiral(a,b):
    return (a**2+b**2)**0.5*np.array([cos((a**2+b**2)),sin((a**2+b**2))])
path_to_follow = mat_reading(lambda a,b : 5+14*np.array([cos(a),sin(0.9*b)]))  # The path the robot has to follow
x0 = np.array([px0, py0, u0, v0, s0, theta0, dtheta0]) # (x,y,vu,vv,s,theta_m,dtheta_m)
path = [x0[0:2]]  # Red dots on the screen, path that the robot follows


key = np.array([0, 0, 0])  # To control the robot using the keyboard
x = x0

draw, end_of_path = 1, 0
# joy = joystick.XboxController() #To control the robot using a game controller

for t in np.arange(0, 1000, dt):
    if end_of_path:
        print("End of simulation")
        break
    # Drawing and screen update
    if draw and (t/dt) % 15 == 0 and t!=0:
        X = x[0:2]
        Vt = x[2:4]
        theta_m, dtheta_m = x[5:7]
        s = x[4]
        ax.clear()
        ax.set_xlim(-w_size-w_shift, w_size-w_shift)
        ax.set_ylim(-w_size, w_size)
        show_info(ax, path_to_follow, X, Vt, theta_m, u, [0,alpha2,alpha3, 0.5, 0.25], [
                  s, t], path, [w_size, w_shift], forces=True, speed=True)
        draw_crab(X, theta_m, ax, 0.5, 0.25)
        if (t/dt) % 45 == 0:
            path.append(X)
        plt.pause(10**-10)
    if keyboard.is_pressed("space"): # To break stop the simulation using the "space" key
        print("End of simulation")
        break
    # Update of the state of the simulation
    t_break = t
    T.append(t)
    u=controller_Non_Col(x)
    state_info.append(x)
    #To control the robot using a game controller
    # turn_left,turn_right,forward,backwards,A_button=joy.read()
    # k2=100
    # k1=100
    # u[1:3]=np.array([k1*(forward-backwards)+k2*turn_left,-k1*(forward-backwards)+k2*turn_left])
    x = rungeKutta2(x, u, dt, state)
    x[4] = max(0, x[4])
    x[5] = sawtooth(x[5])

#To plot data after the simulation
# ax.clear()
# state_info=np.array(state_info)
# data=state_info[:,5]
# ax.plot(T,data, color='red')
# ax.set_xlabel('time (s)')
# ax.set_ylabel('data')
# plt.show()