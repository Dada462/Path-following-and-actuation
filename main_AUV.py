from cmath import atan
from platform import mac_ver
from numpy import cos, real, sin, tanh, arctan, arctan2, pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2, sawtooth, R, path_info_update, draw_crab, show_info,mat_reading,R3
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


def controller(x):
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
    # ddbeta_mes = (-u/(nu**2*mv)*(mur*u*a+dd_v)-2*u*dnu * dv/(nu**3))/(1-(cos(beta_mes)**2)*mur/mv)

    ddtheta_m = dddelta-ddbeta+ddtheta_c + (k1+k3)*(ddelta-dpsi)+(k5+k3*k1)*(delta-psi)

    F=A_plus@np.array([(d_u+2*(1-u)*mu),0,mr*ddtheta_m+d_r])
    return np.array([*F,ds])

def controller_2(x):
    global end_of_path
    X = x[0:2]
    Vt = x[2:4]
    s = x[4]
    theta_m, dtheta_m = x[5:7]
    u, v = Vt
    nu = (u**2+v**2)**0.5

    d_v = -Yv*u*v-Yvabsv*v*abs(v)
    d_u=-Xuu*u**2-Xvv*v**2
    d_r=-Nv*u*v-Nvabsv*v*abs(v)-Nr*u*dtheta_m
    beta = arctan2(v, u)


    F = path_info_update(path_to_follow, s)
    theta_c = F.psi
    s1, y1 = R(theta_c).T@(X-F.X)
    theta = sawtooth(theta_m-theta_c)
    psi = sawtooth(theta+beta)
    ks = 1
    ds = cos(psi)*nu + ks*s1
    dtheta_c = F.C_c*ds
    ds1, dy1 = R(theta)@Vt-ds*np.array([1-F.C_c*y1, F.C_c*s1])

    Kdy1 = 1
    psi_a=pi/2
    delta = -psi_a*tanh(Kdy1*y1)
    ddelta = -psi_a*Kdy1*(1-tanh(Kdy1*y1)**2)*dy1
    dpsi=ddelta+2*(delta-psi)
    
    theta_m_d=-pi/2
    dtheta_m_d=0
    ddtheta_m_d=0
    ddtheta_m=ddtheta_m_d+2*(dtheta_m_d-dtheta_m)+1*(theta_m_d-theta_m)
    dnu=2*(1-nu)
    dbeta=dpsi-dtheta_m+dtheta_c
    
    C=np.array([[cos(beta)/mu,sin(beta)/mv,0],[-sin(beta)/mu,cos(beta)/mv,mur*u/(Nr*mv)],[0,0,1/mr]])
    d1_r=Nv*u*v+Nvabsv*v*abs(v)
    b=np.array([-cos(beta)*d_u/mu-sin(beta)*(d_v-mur*u*dtheta_m)/mv,-u*d_v/(mv*nu**2)+v*d_u/(mu*nu**2) +mur*u*(d1_r-mr*ddtheta_m)/(Nr*mv*nu**2),-d_r/mr])
    Fu,Fv,Gamma=np.linalg.pinv(C)@(np.array([dnu,dbeta,ddtheta_m])-b)
    F=A_plus@np.array([Fu,Fv,Gamma])
    return np.array([*F,ds])

def controller_1(x):
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
    dpsi=ddelta+2*(delta-psi)
    dbeta=dpsi-dtheta_m+dtheta_c
    theta_m_d=-pi/2
    dtheta_m_d=0
    ddtheta_m_d=0
    ddtheta_m=ddtheta_m_d+2*(dtheta_m_d-dtheta_m)+1*(theta_m_d-theta_m)
    d=np.array([d_u/mu,(d_v+mur*u*dtheta_m)/mv,d_r/mr])
    M=np.array([[mu,0,0],[0,mv,0],[0,0,mr]])
    Fu,Fv,Gamma=M@(R3(beta)@np.array([dnu,nu*dbeta,ddtheta_m])+d)
    F=A_plus@np.array([Fu,Fv,Gamma])
    return np.array([*F,ds])


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


alpha2=pi/2
alpha3=-pi/2
d1=d2=d3=1
d1=0
L=1
r=0.25
D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
A=np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-L,-L,-L]])@D
A_plus=np.linalg.pinv(A) #penrose inverse


# Drawing and window info
dt, w_size, w_shift = 0.01, 15, -7
fig, ax = plt.subplots(figsize=(8, 7))

# Lists to stock the information. For viewing resulsts after the simulation
# T is the list temporal values, state_info[i] is the state x(t_i) where t_i=T[i]
T, state_info = [], []
t_break = 0  # it will be the time at which the simulation stops

# Initial conditions
px0, py0 = -5, 10
u0, v0 = -5,0
theta0, dtheta0 = 0, 0
s0 = 0
beta0=arctan2(v0,u0)


f1_0, f2_0, f3_0 = 0, 0, 0
ds0 = 0

path_to_follow = mat_reading()  # The path the robot has to follow
x0 = np.array([px0, py0, u0, v0, s0, theta0, dtheta0]) # (x,y,vu,vv,s,theta_m,dtheta_m)
u0 = np.array([f1_0, f2_0, f3_0,ds0])  # (f1,f2,f3)
path = [x0[0:2]]  # Red dots on the screen, path that the robot follows


key = np.array([0, 0, 0])  # To control the robot using the keyboard
x = x0
u = u0

draw, end_of_path = 1, 0
# joy = joystick.XboxController()

for t in np.arange(0, 1000, dt):
    if end_of_path:
        print("End of simulation")
        break
    # Drawing and screen update
    if draw and (t/dt) % 15 == 0:
        X = x[0:2]
        Vt = x[2:4]
        theta_m, dtheta_m = x[5:7]
        s = x[4]
        ax.clear()
        ax.set_xlim(-w_size-w_shift, w_size-w_shift)
        ax.set_ylim(-w_size, w_size)
        show_info(ax, path_to_follow, X, Vt, theta_m, u, [0,alpha2,alpha3, 0.5, 0.25], [
                  s, t], path, [w_size, w_shift], forces=False, speed=True)
        draw_crab(X, theta_m, ax, 0.5, 0.25)
        ax.text(-w_size/2, w_size+0.5, 'u:' +str(round(x[2],2)), style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
        if (t/dt) % 45 == 0:  # Change the value to change the frequency of the red dots of the path
            path.append(X)
        plt.pause(10**-10)
    if keyboard.is_pressed("space"):
        break
    # Update of the state of the simulation
    t_break = t
    T.append(t)
    state_info.append(x)
    # if t>20:
    #     if t<20.1:
    #         d1=0
    #         D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
    #         A=np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-L,-L,-L]])@D
    #         A_plus=np.linalg.pinv(A) #penrose inverse
    #     u=controller(x)
    # else:
    #     u=controller_1(x)
    u=controller(x)
    # turn_left,turn_right,forward,backwards,A_button=joy.read()
    # k2=100
    # k1=100
    # u[1:3]=np.array([k1*(forward-backwards)+k2*turn_left,-k1*(forward-backwards)+k2*turn_left])
    x = rungeKutta2(x, u, dt, state)
    x[4] = max(0, x[4])
    # x[5] = sawtooth(x[5])

ax.clear()
state_info=np.array(state_info)
s=state_info[:,4]
data=state_info[:,5]
ax.plot(s,data, color='red')
ax.set_xlabel('time (s)')
ax.set_ylabel('data')
plt.show()