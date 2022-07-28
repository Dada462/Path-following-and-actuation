from cmath import atan
from numpy import cos,sin,tanh,arctan,arctan2,pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2,sawtooth,mat_reading,R,path_info_update,draw_crab,show_info,key_press,R3,path_data,find
import joystick




def mat_reading_1():
    def sum(X):
        s = 0
        if len(X)!=0 and len(X)!=1:
            for x in X:
                s += x
        return s
    n=5000
    n=n+2
    def f(X,Y):
        return 5+9*np.array([cos(X),sin(0.9*Y)])
    X=np.linspace(-10,10,n)
    Y=np.linspace(-10,10,n)
    X,Y=f(X,Y)
    ds=np.array([((X[i+1]-X[i])**2+(Y[i+1]-Y[i])**2)**0.5 for i in range(0,n-1)])
    S=np.array([sum(ds[0:i]) for i in range(n)])

    dX=np.array([X[i+1]-X[i] for i in range(0,n-1)])
    dY=np.array([Y[i+1]-Y[i] for i in range(0,n-1)])
    
    psi=arctan2(dY,dX)
    dpsi=np.array([psi[i+1]-psi[i] for i in range(0,n-2)])
    ds=list(ds)
    ds.pop()
    ds=np.array(ds)
    C=dpsi/ds

    X=list(X)
    X.pop()
    X.pop()
    X=np.array(X)
    
    Y=list(Y)
    Y.pop()
    Y.pop()
    Y=np.array(Y)

    psi=list(psi)
    psi.pop()
    psi=np.array(psi)

    S=list(S)
    S.pop()
    S.pop()
    S=np.array(S)

    XY=np.array([X,Y]).T
    print(np.shape(XY),np.shape(psi),np.shape(S),np.shape(C))
    path_to_follow = path_data(XY, psi, S, C)
    return path_to_follow

def state(X,controller):
    _,_,vtx,vty,s,theta_m,dtheta_m=X.flatten()
    u=controller[0:-1]
    ds=controller[-1]
    Vt=np.array([vtx,vty])
    b=np.vstack((-dtheta_m*(R(pi/2)@Vt).reshape((2,1)),0)).flatten()
    dVt_and_ddtheta_m=(A@u+b).flatten()
    dVt=dVt_and_ddtheta_m[0:2]
    ddtheta_m=dVt_and_ddtheta_m[2]
    return np.hstack((R(theta_m)@Vt,dVt,ds,dtheta_m,ddtheta_m))

def dynamic_controller2(x,dVt):
    global end_of_path
    X=x[0:2]
    Vt=x[2:4]
    s=x[4]
    theta_m,dtheta_m=x[5:7]
    u,v=Vt
    nu=(u**2+v**2)**0.5
    if nu<0.5:
        return np.array([-1,1,-1,0])
    dnu=dVt.T@Vt/nu
            
    F = path_info_update(path_to_follow,s)
    if F==None:
        end_of_path=1
        return np.array([0,0,0,0])
    theta_c=F.psi
    theta = theta_m-theta_c
    beta=arctan2(Vt[1],Vt[0])
    psi=sawtooth(theta+beta)

    s1,y1 = R(theta_c).T@(X-F.X)
    
    ks=3
    ds=cos(psi)*nu +ks*s1
    dtheta_c = F.C_c*ds
    ds1,dy1= R(psi)@np.array([nu,0])-ds*np.array([1-F.C_c*y1,F.C_c*s1])
    dtheta=dtheta_m-dtheta_c


    Kdy1=5
    delta = -pi/2*tanh(Kdy1*y1*nu)
    ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*nu)**2)*(dy1*nu+dnu*dy1)
    
    gamma = 0.5
    k2=1
    dpsi=ddelta - k2*sawtooth(psi-delta)
    vtd=(1-nu)
    dbeta=dpsi-dtheta
    ddtheta_m=-dtheta_m+(pi/4-theta_m)
    W=np.array([vtd,dbeta,ddtheta_m])

    B=np.array([[1,0,0],[0,1/nu,0],[0,0,1]])@R3(beta).T@A
    F=np.linalg.pinv(B)@(W+np.array([0,dtheta_m,ddtheta_m]))

    return np.array([F[0],F[1],F[2],ds])

#Geometric parameters of the robot
alpha2=pi/2
alpha3=-pi/2
d1=d2=d3=1
# d1=0
L=1
r=0.25
m=1
J=1
D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
A=1/m*np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-m*L/J,-m*L/J,-m*L/J]])@D
A_plus=np.linalg.pinv(A) #penrose inverse


#Drawing and window info
dt,w_size,w_shift= 0.01,15,-7
fig,ax=plt.subplots(figsize=(8,7))

#Lists to stock the information. For viewing resulsts after the simulation
T,state_info=[],[] #T is the list temporal values, state_info[i] is the state x(t_i) where t_i=T[i]
t_break=0 #it will be the time at which the simulation stops

#Initial conditions
px0,py0=-5,10
u0,v0=0.1,0
theta0,dtheta0=0,0
s0=0

f1_0,f2_0,f3_0=0,0,0
ds0=0

path_to_follow=mat_reading_1() #The path the robot has to follow
x0=np.array([px0,py0,u0,v0,s0,theta0,dtheta0]) # (x,y,vu,vv,theta_m,dtheta_m)
u0=np.array([f1_0,f2_0,f3_0,ds0]) # (f1,f2,f3)
path=[x0[0:2]] #Red dots on the screen, path that the robot follows

key=np.array([0,0,0]) #To control the robot using the keyboard
x=x0
u=u0

draw,end_of_path=1,0

joy = joystick.XboxController()

for t in np.arange(0,100,dt):
    if end_of_path:
        print("End of simulation")
        break
    #Drawing and screen update
    X=x[0:2]
    Vt=x[2:4]
    theta_m,dtheta_m=x[5:7]

    if draw and (t/dt)%30==0:
        s=x[4]
        ax.clear()
        ax.set_xlim(-w_size-w_shift,w_size-w_shift)
        ax.set_ylim(-w_size,w_size)
        show_info(ax,path_to_follow,X,Vt,theta_m,u,[0,alpha2,alpha3,0.5,r],[s,t],path,[w_size,w_shift],forces=True,speed=False)     
        draw_crab(X,theta_m,ax,0.5,r)
        if (t/dt)%30==0: #Change the value to change the frequency of the red dots of the path
            path.append(X)
        plt.pause(10**-10)
    
    #Update of the state of the simulation
    t_break=t
    T.append(t)
    dVt=state(x,u)[2:4]
    state_info.append(x)
    u=dynamic_controller2(x,dVt)
    u=20*tanh(0.2*u)
    x=rungeKutta2(x,u,dt,state)
    x[4]=max(0,x[4])
    x[5]=sawtooth(x[5])


ax.clear()
state_info=np.array(state_info)
ax.plot(state_info[:,3],state_info[:,5], color='red')
ax.set_xlabel('time (s)')
ax.set_ylabel('data')
plt.show()