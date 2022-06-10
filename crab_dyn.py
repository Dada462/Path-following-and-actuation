from cmath import atan
from numpy import cos,sin,tanh,arctan,pi
import scipy.io
import numpy as np
import keyboard
import matplotlib.pyplot as plt



def rungeKutta2(x,u,h,f):
    x = x + h*(0.5*f(x,u) + 0.5*f(x + h*f(x,u),u))
    return x

def sawtooth(x):
    return (x+pi)%(2*pi)-pi   # or equivalently   2*arctan(tan(x/2))

class path_data:
    def __init__(self,X,psi,s,C_c):
        self.s = s
        self.psi = psi
        self.C_c = C_c
        self.X = X
    def __str__(self):
        return str(self.s) + ' ' + str(self.psi) + ' ' + str(self.C_c) + ' ' + str(self.X)

def mat_reading(path='PATH.mat'): #Return the path for follow info: s,X,psi,C_c
    mat = scipy.io.loadmat(path)
    
    s=[]
    for i in range(len(mat['Chemin']['s'][0,:])):
        s.append(mat['Chemin']['s'][0,:][i][0][0])
    s=np.array(s)

    X=np.array([[],[]])
    for i in range(len(mat['Chemin']['X'][0,:])):
        X=np.hstack((X,mat['Chemin']['X'][0,:][i]))
    X=X.T

    psi=[]
    for i in range(len(mat['Chemin']['psi'][0,:])):
        psi.append(mat['Chemin']['psi'][0,:][i][0][0])
    psi=np.array(psi)

    C_c=[]
    for i in range(len(mat['Chemin']['C_c'][0,:])):
        C_c.append(mat['Chemin']['C_c'][0,:][i][0][0])
    C_c=np.array(C_c)
    path_to_follow=path_data(X,psi,s,C_c)
    return path_to_follow
path_to_follow=mat_reading()

def find(elements,s):
    inds = [i for (i, val) in enumerate(elements) if val >s]
    if inds!=[]:
        return inds[0]
    else:
        return "None"

def path_info_update(path_info_last,s):
    I=find((path_info_last.s),s)
    path_info=path_data(0,0,0,0)
    if I=='None':
        print('End of the path')
        return None
    else:
        Delta_S_Lievre = (path_info_last.s)[I]-(path_info_last.s)[I-1]
        ratio_S_Lievre = (s-path_info_last.s[I-1])/Delta_S_Lievre
        path_info.s = s
        path_info.psi = path_info_last.psi[I-1]*(1-ratio_S_Lievre) + path_info_last.psi[I]*(ratio_S_Lievre)
        path_info.C_c = path_info_last.C_c[I-1]*(1-ratio_S_Lievre) + path_info_last.C_c[I]*(ratio_S_Lievre)
        path_info.X = path_info_last.X[I-1]*(1-ratio_S_Lievre) + path_info_last.X[I]*(ratio_S_Lievre)
        return path_info


alpha2=pi/2
alpha3=-pi/2
d1=d2=d3=1
d1=0
def state(X,u):
    _,_,vtx,vty,theta_m,dtheta_m=X.flatten()
    Vt=np.array([vtx,vty])

    D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
    A=1/m*np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-r/J,-r/J,-r/J]])@D
    b=np.vstack((-dtheta_m*(R(pi/2)@Vt).reshape((2,1)),0)).flatten()
    
    dVt_and_ddtheta_m=(A@u+b).flatten()
    dVt=dVt_and_ddtheta_m[0:2]
    ddtheta_m=dVt_and_ddtheta_m[2]
    return np.hstack((R(theta_m)@Vt,dVt,dtheta_m,ddtheta_m))

def R(theta):
    return np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])

L,r=0.5,0.25
m=1
J=1
def draw_crab(P,theta):
    theta_deg=180*theta/pi-90
    wheel_size=0.5
    body=plt.Circle(P,L,edgecolor='#1F6EE1',facecolor='none')
    wheel1 = plt.Rectangle(P+R(theta-pi/2)@(L*np.array([0,1])-np.array([r/2,0])), r, wheel_size,angle=theta_deg, linewidth=1,edgecolor='g',facecolor='none')
    wheel2 = plt.Rectangle(P+R(theta-pi/2+2*pi/3)@(L*np.array([0,1])-np.array([r/2,0])), r, wheel_size, linewidth=1,angle=theta_deg+120,edgecolor='r',facecolor='none')
    wheel3 = plt.Rectangle(P+R(theta-pi/2-2*pi/3)@(L*np.array([0,1])-np.array([r/2,0])), r, wheel_size, linewidth=1,angle=theta_deg-120,edgecolor='r',facecolor='none')
    
    ax.add_patch(body)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)
    ax.add_patch(wheel3)

t0=5
def sat(t):
    # return (1+np.tanh(4*(t0-t)))/2
    if t>t0:
        return 0
    else:
        return 1

def kin_controller(x,path_to_follow):
    global end_of_path
    Vt=x[2:4]
    vu,vv=Vt[0:2]

    X,theta_m=x[0:2],x[4]
    M=np.array([[cos(theta_m),-sin(theta_m),0],[sin(theta_m),cos(theta_m),0],[0,0,1]])
    F = path_info_update(path_to_follow,s)
    if F==None:
        end_of_path=1
        return np.array([0,0,0]),0
    theta_c=F.psi
    R = np.array([[cos(theta_c),-sin(theta_c)],[sin(theta_c),cos(theta_c)]])
    s1,y1 = (R.T)@(X-F.X)
    theta = theta_m-theta_c
    K_ds=5
    ds = vu*cos(theta)-sin(theta)*vv+K_ds*s1
    
    dX_F =  np.array([ds,0])
    ds1,dy1= (M[0:2,0:2]@R.T)@(Vt[0:2])-dX_F
    dpsi_F = F.C_c*ds

    #delta
    nu=(Vt[0]**2+Vt[1]**2)**0.5
    dnu=1
    Kdy1=1
    delta = -pi/2*tanh(Kdy1*y1*nu)
    ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*nu)**2)*dy1*dnu
    
    K=4
    gamma=0.5
    # y0=0.5
    # vv=-2*tanh((theta-delta))**2*cos(theta)*tanh(y1/y0)
    if theta-delta!=0:
        dpsi = dpsi_F + (ddelta -K*(theta - delta) -gamma*y1*nu*(sin(theta)-sin(delta))/sawtooth(theta-delta))
    else:
        dpsi = dpsi_F + (ddelta -K*(theta - delta) -gamma*y1*(vu +vv))
    u = np.array([vu,vv,dpsi])
    return u,ds

def dynamic_controller(x,u_kin):
    Vt=x[2:4]
    dtheta_m=x[5]
    D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
    A=1/m*np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-r/J,-r/J,-r/J]])@D
    b=np.vstack((-dtheta_m*(R(pi/2)@Vt).reshape((2,1)),0)).flatten()
    A_plus=np.linalg.pinv(A) #penrose inverse
    k=0*(np.eye(3)-A_plus@A)@D@np.array([1,1,1])
    u=A_plus@(u_kin-b) + k
    return u

def dynamic_controller2(x,s):
    global end_of_path
    Vt=x[2:4]
    dtheta_m=x[5]

    X,theta_m=x[0:2],x[4]
    M=np.array([[cos(theta_m),-sin(theta_m),0],[sin(theta_m),cos(theta_m),0],[0,0,1]])
    F = path_info_update(path_to_follow,s)
    if F==None:
        end_of_path=1
        return np.array([0,0,0]),0
    theta_c=F.psi
    R_c = np.array([[cos(theta_c),-sin(theta_c)],[sin(theta_c),cos(theta_c)]])
    s1,y1 = (R_c.T)@(X-F.X)
    theta = theta_m-theta_c
    ks=3
    ds=np.dot(R(theta)[0,:],Vt) + ks*s1
    
    ds1,dy1= (M[0:2,0:2]@R_c.T)@(Vt[0:2])-np.array([ds,0])
    dtheta_c = F.C_c*ds
    dtheta=dtheta_m-dtheta_c
    # dds1,ddy1=R(theta)@(-theta_c/m*A[0:2,:]@u_last - np.array([dds,]))

    #delta
    Kdy1=3
    nu=3
    delta = -pi/2*tanh(Kdy1*y1*nu)
    ddelta = -pi/2*Kdy1*nu*(1-tanh(Kdy1*y1*nu)**2)*dy1

    Vtd=R(delta-theta)@np.array([nu,0])
    dVtd=R(delta-theta)@np.array([0,(ddelta-dtheta)*nu])
    
    D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
    A=1/m*np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-r/J,-r/J,-r/J]])@D
    b=np.vstack((-dtheta_m*(R(pi/2)@Vt).reshape((2,1)),0)).flatten()
    A_plus=np.linalg.pinv(A) #penrose inverse

    # k0,k1=1,1
    # ddtheta=k0*tanh(dy1)-k1*tanh(dy1)+1.5*tanh(s1)+2*tanh(ds1)
    # beta=np.arctan2(*Vt)
    # ddtheta=-8*dtheta_m+16*sawtooth(-theta_m)
    ddtheta=0
    k=10
    ddVt=(dVtd -y1*np.array([sin(theta),cos(theta)])-k*(Vt-Vtd)).reshape((2,1))
    F=A_plus@(np.vstack((ddVt,-dtheta_m)).flatten()-b)
    u=F.flatten()

    return u,ds

def key_press():
    global key,u_kin,end_of_path
    alpha=5
    if keyboard.is_pressed("up"):
        key[0]=alpha
    elif keyboard.is_pressed("down"):
        key[0]=-alpha
    else:
        key[0]=0
    if keyboard.is_pressed("left"):
        key[1]=alpha
    elif keyboard.is_pressed("right"):
        key[1]=-alpha
    else:
        key[1]=0
    
    if keyboard.is_pressed("e"):
        key[2]=alpha
    elif keyboard.is_pressed("d"):
        key[2]=-alpha
    else:
        key[2]=0
    if keyboard.is_pressed("space"):
        end_of_path=1
    u_kin=key

#Drawing and window info
dt,w_size,w_shift= 0.01,15,-7
fig,ax=plt.subplots(figsize=(8,7))
#T is the list temporal values, state_info[i] is the state x(t_i) where t_i=T[i]
T,state_info=[],[]
t_break=0 #it will be the time at which the simulation stops
vu,vv=1,0 #initial speed values along the u and v axis
x0=np.array([-5,10,vu,vv,0,0]) # (x,y,vu,vv,theta_m,dtheta_m)
u0=np.array([0,0,0]) # (f1,f2,f3)
key=np.array([0,0,0])
path=[x0[0:2]]#Red dots on the screen, path that the robot follows
x=x0
u=u0
s=0
draw,end_of_path=1,0
ds=0
Vtd=np.array([1,1])

for t in np.arange(0,30,dt):
    # u_kin,ds=kin_controller(x,path_to_follow)
    u,ds=dynamic_controller2(x,s)
    key_press()

    if end_of_path:
        print("End of simulation")
        break 
    if s<0:
        s=0
    #Drawing and screen update
    if draw and (t/dt)%5==0:
        ax.clear()
        ax.set_xlim(-w_size-w_shift,w_size-w_shift)
        ax.set_ylim(-w_size,w_size)
        if t>t0:
            ax.text(w_size, w_size+0.5,'MOTOR DEAD', style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
        ax.plot(*path_to_follow.X.T,c='#3486F4')
        draw_crab(x[0:2],x[4])
        ax.quiver(*(x[0:2]),*R(x[4])@x[2:4],color='red',scale=10)
        
        #To plot foces
        # ax.quiver(*(x[0:2]+R(x[4])@R(0)@np.array([L,0])),*R(x[4]-pi/2)@np.array([u[0],0]),color='red',scale=10)
        # ax.quiver(*(x[0:2]+R(x[4])@R(alpha2)@np.array([L,0])),*R(x[4]-pi/2+alpha2)@np.array([u[1],0]),color='red',scale=10)
        # ax.quiver(*(x[0:2]+R(x[4])@R(alpha3)@np.array([L,0])),*R(x[4]-pi/2+alpha3)@np.array([u[2],0]),color='red',scale=10)
        
        ax.scatter(*path_info_update(path_to_follow,s).X,c='#34F44C')
        if (t/dt)%30==0: #Change the value to change the frequency of the red dots of the path
            path.append(x[0:2])

        for point in path:
            ax.scatter(*point,c='red',s=10)
        info='u1 : '+str(round(u[0],2))+'\n'+'u2 : '+str(round(u[1],2)) +'\n'+'u3 : '+str(round(u[2],2))
        ax.text(w_size/2, w_size+0.5,info, style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
        ax.text(0, w_size+1, 'time :' + str(round(t,2)) +' s', style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})    
        plt.pause(10**-10)
    #Update of the state of the simulation
    T.append(t)
    state_info.append(x)
    x=rungeKutta2(x,u,dt,state)
    x[4]=sawtooth(x[4])
    s=s+ds*dt
    t_break=t


# Plotting results
# ax.clear()
# data1=np.array(state_info)[:,2]
# ax.plot(T,data1)
# T=np.array(T)
# w0=x0[5]
# ax.plot(T,cos(w0*T)*x0[2]+x0[3]*sin(w0*T)+1/w0*0.5/m)
# ax.set_xlabel('time (s)')
# ax.set_ylabel('data')
# plt.show()