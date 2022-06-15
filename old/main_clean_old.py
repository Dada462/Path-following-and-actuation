from cmath import atan
from numpy import cos,sin,tanh,arctan,pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2,sawtooth,mat_reading,R,path_info_update,draw_crab,show_info,key_press


def state(X,u):
    _,_,vtx,vty,theta_m,dtheta_m=X.flatten()
    Vt=np.array([vtx,vty])
    b=np.vstack((-dtheta_m*(R(pi/2)@Vt).reshape((2,1)),0)).flatten()
    dVt_and_ddtheta_m=(A@u+b).flatten()
    dVt=dVt_and_ddtheta_m[0:2]
    ddtheta_m=dVt_and_ddtheta_m[2]
    return np.hstack((R(theta_m)@Vt,dVt,dtheta_m,ddtheta_m))

def dynamic_controller2(x,s,s_last):
    global end_of_path
    Vt=x[2:4]
    dtheta_m=x[5]

    X,theta_m=x[0:2],x[4]
    M=np.array([[cos(theta_m),-sin(theta_m),0],[sin(theta_m),cos(theta_m),0],[0,0,1]])
    F_last = path_info_update(path_to_follow,s_last)
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
    ds_last=(s-s_last)/dt
    dds=(ds-ds_last)/dt
    ds1,dy1= (M[0:2,0:2]@R_c.T)@Vt-ds*np.array([1-F.C_c*y1,F.C_c*s1])
    dtheta_c = F.C_c*ds
    ddtheta_c = (F.C_c-F_last.C_c)/dt
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

    k=10
    dVt=(dVtd -y1*np.array([sin(theta),cos(theta)])-k*(Vt-Vtd)).reshape((2,1))
    F=A_plus@(np.vstack((dVt[0],dVt[1],0)).flatten()-b)
    u=F.flatten()

    return u,ds



#Geometric parameters of the robot
alpha2=pi/2
alpha3=-pi/2
d1=d2=d3=1
L=0.5
r=0.25
m=1
J=1
D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
A=1/m*np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-r/J,-r/J,-r/J]])@D




#Drawing and window info
dt,w_size,w_shift= 0.01,15,-7
fig,ax=plt.subplots(figsize=(8,7))

#Lists to stock the information. For viewing resulsts after the simulation
T,state_info=[],[] #T is the list temporal values, state_info[i] is the state x(t_i) where t_i=T[i]
t_break=0 #it will be the time at which the simulation stops

#Initial conditions
px0,py0,u0,v0,theta0,dtheta0=-5,10,1,0,0,0
f1_0,f2_0,f3_0=0,0,0
path_to_follow=mat_reading() #The path the robot has to follow
x0=np.array([px0,py0,u0,v0,theta0,dtheta0]) # (x,y,vu,vv,theta_m,dtheta_m)
u0=np.array([f1_0,f2_0,f3_0]) # (f1,f2,f3)
path=[x0[0:2]] #Red dots on the screen, path that the robot follows

key=np.array([0,0,0]) #To control the robot using the keyboard
x=x0
u=u0
s=0
ds=0
draw,end_of_path=1,0

for t in np.arange(0,30,dt):
    s_last=s
    s=s+ds*dt

    if end_of_path:
        print("End of simulation")
        break 
    
    #Drawing and screen update
    X=x[0:2]
    Vt=x[2:4]
    theta_m,dtheta_m=x[4:6]
    if draw and (t/dt)%5==0:
        ax.clear()
        ax.set_xlim(-w_size-w_shift,w_size-w_shift)
        ax.set_ylim(-w_size,w_size)
        show_info(ax,path_to_follow,X,Vt,theta_m,u,[0,alpha2,alpha3,L,r],[s,t],path,[w_size,w_shift],forces=True,speed=False)     
        draw_crab(X,theta_m,ax,L,r)
        if (t/dt)%30==0: #Change the value to change the frequency of the red dots of the path
            path.append(X)
        plt.pause(10**-10)
    #Update of the state of the simulation
    T.append(t)
    state_info.append(x)
    x=rungeKutta2(x,u,dt,state)
    x[4]=sawtooth(x[4])
    s=max(0,s)
    t_break=t
    ds_last=ds
    u,ds=dynamic_controller2(x,s,ds_last)
