from cmath import atan
from platform import mac_ver
from numpy import cos, real, sin, tanh, arctan, arctan2, pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2, sawtooth, R, path_info_update, draw_crab, show_info,mat_reading,R3,colors,color
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
    ddbeta = (-u/(nu**2*mv)*(mur*u*zeta+dd_v+mur*du*dtheta_m)-2*dnu*(u*dv-du*v)/(nu**3))/(1-(cos(beta_mes)**2)*mur/mv)

    ddtheta_m = dddelta-ddbeta+ddtheta_c + (k1+k3)*(ddelta-dpsi)+(k5+k3*k1)*(delta-psi)

    F=A_plus@np.array([(d_u+2*(1-u)*mu),0,mr*ddtheta_m+d_r])
    return np.array([*F,ds]),ddtheta_m

def controller_1(x):
    #Has a singularity when two motors are colinear
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

    dnu=2*(0.5-nu)
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


# Drawing and window info
dt, w_size, w_shift = 0.1, 15, -7
fig, ax = plt.subplots(figsize=(8, 7))

ALPHA=[(0,pi/2),(2*pi/3,-2*pi/3),(pi/3,2*pi/3)]
for j in range(3):
    alpha2,alpha3=ALPHA[j]
    L=1
    r=0.25
    d1=d2=d3=1
    d2=0
    D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
    A=np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-L,-L,-L]])@D
    A_plus=np.linalg.pinv(A) #penrose inverse

    STATES=[]
    Rmin=-10
    Rmax=11
    dR=2.5

    for Radius in np.arange(Rmin,Rmax,dR):
        # Lists to stock the information. For viewing resulsts after the simulation
        # T is the list temporal values, state_info[i] is the state x(t_i) where t_i=T[i]
        T, state_info = [], []
        t_break = 0  # it will be the time at which the simulation stops

        # Initial conditions
        px0, py0 = -5, 10
        u0, v0 = .5,0
        theta0, dtheta0 = 0, 0
        s0 = 0
        #lambda a,b : 5+Radius*np.array([cos(a),sin(b)])
        path_to_follow = mat_reading(lambda a,b :5+Radius*np.array([cos(a),sin(b)]))  # The path the robot has to follow
        x0 = np.array([px0, py0, u0, v0, s0, theta0, dtheta0]) # (x,y,vu,vv,s,theta_m,dtheta_m)
        path = [x0[0:2]]  # Red dots on the screen, path that the robot follows


        key = np.array([0, 0, 0])  # To control the robot using the keyboard
        x = x0

        end_of_path = 0
        Tmax=250
        for t in np.arange(0, Tmax, dt):
            if end_of_path:
                print("End of simulation")
                break
            # Update of the state of the simulation
            t_break = t
            T.append(t)
            u=controller_1(x)
            d_r=-Nv*x[2]*x[3]-Nvabsv*x[3]*abs(x[3])-Nr*x[2]*x[5]
            Gamma=(A@u[0:-1])[-1]
            ddtheta_m=1/mr*(Gamma-d_r)
            # x[6]-0*u[-1]/Radius
            state_info.append(np.hstack((x,0)))
            x = rungeKutta2(x, u, dt, state)
            x[4] = max(0, x[4])
        STATES.append(state_info)

    ax.clear()
    i=0
    x=colors()
    C=[]
    for c in x:
        C.append(c.split(',')[-1])

    M=0
    for state_info in STATES:
        data=np.array(state_info)
        ax.plot(T,data[:,6],label='Slope='+str(Rmin+dR*i),color=color(C,Rmin,Rmax,Rmin+dR*i))
        M=max(M,max((data[:,6])))
        # ax.plot(T,data[:,4]/30,label='θ',color='c')
        # ax.plot(T,2*pi/3*np.ones(np.shape(T)),label='θ',color='c')
        # ax.plot(T,arctan2(data[:,3],data[:,2]),label='θ',color='c')
        i+=1
    def get_sub(x):
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
        res = x.maketrans(''.join(normal), ''.join(sub_s))
        return x.translate(res)
    print(M)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('dθ'+get_sub('m')+'/dt')
    ax.text(Tmax*2/3,3/4*M, '(β'+get_sub('2')+',β'+get_sub('3')+')=(' +str(round(alpha2,2)) +' rd, '+str(round(alpha3,2))+' rd)', style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
    plt.title('Behaviour of ' + 'dθ'+get_sub('m')+'/dt '+ 'for linear paths')
    plt.legend(loc='lower right')
    plt.savefig("linear_paths_2NC_"+str(j+1)+".png")
    # plt.show()