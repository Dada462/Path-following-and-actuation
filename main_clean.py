from cmath import atan
from numpy import cos,sin,tanh,arctan,pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2,sawtooth,mat_reading,R,path_info_update,draw_crab,show_info,key_press


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

def dynamic_controller2(x,last_values):
    global end_of_path
    X=x[0:2]
    Vt=x[2:4]
    s=x[4]
    theta_m,dtheta_m=x[5:7]

    F = path_info_update(path_to_follow,s)
    if F==None:
        # end_of_path=1
        F=last_values[-2]
        return np.array([0,0,0,0])
    theta_c=F.psi
    theta = theta_m-theta_c

    s1,y1 = R(theta_c).T@(X-F.X)
    
    ks=3
    ds=np.dot(R(theta)[0,:],Vt) + ks*s1
    dtheta_c = F.C_c*ds
    ds1,dy1= R(theta)@Vt-ds*np.array([1-F.C_c*y1,F.C_c*s1])
    dtheta=dtheta_m-dtheta_c


    Kdy1=3
    nu=1
    dnu=0
    # delta = -pi/2*tanh(Kdy1*y1*nu)
    # ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*nu)**2)*(dy1*nu+dnu*dy1)
    
    Vt_last,dY_last,dvr_last,F_last,ds_last=last_values
    dtheta_c_last=F_last.C_c*ds_last
    dvr=(Vt-Vt_last)/dt
    du,dv=dvr
    ddu,ddv=(dvr-dvr_last)/dt
    dY=np.array([ds1,dy1])
    ddY=(dY-dY_last)/dt
    dds1,ddy1=ddY

    u,v=Vt
    delta = -pi/2*tanh(Kdy1*y1*u)
    ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*u)**2)*(dy1*u+du*dy1)
    dddelta=-pi/2*(Kdy1*(ddu*y1+2*du*dy1 +u*ddy1)*(1-tanh(Kdy1*u*y1)**2) -2*Kdy1**2*(du*y1+u*dy1)**2*(1-tanh(Kdy1*u*y1)**2)*tanh(Kdy1*u*y1))
    gamma = 0.5
    k2=10
    k3=1
    if theta-delta!=0:
        zeta=ddelta-gamma*y1*u*(sin(theta) - sin(delta)) / sawtooth(theta - delta) - k2*sawtooth(theta-delta)
        dzeta=dddelta - gamma*y1*u*((dtheta*cos(theta)-ddelta*cos(delta))*sawtooth(theta-delta)-(dtheta-ddelta)*(sin(theta)-sin(delta)))/(sawtooth(theta-delta)**2)-gamma*(dy1*u+du*y1)*(sin(theta) - sin(delta)) / sawtooth(theta - delta)-k2*(dtheta-ddelta)
        eps=dtheta-zeta
        deps=-1/gamma*sawtooth(theta-delta) - k3*eps
        ddtheta=deps+dzeta
        ddtheta_c=(dtheta_c-dtheta_c_last)/dt
        ddtheta_m=ddtheta+ddtheta_c
    else:
        # zeta=ddelta-gamma*y1*u
        # dzeta=dddelta - gamma*y1*u*((dtheta*cos(theta)-ddelta*cos(delta))*sawtooth(theta-delta)-(dtheta-ddelta)*(sin(theta)-sin(delta)))/(sawtooth(theta-delta)**2)-gamma*(dy1*u+du*y1)-k2*(dtheta-ddelta)
        # eps=dtheta-zeta
        # deps=-1/gamma*sawtooth(theta-delta) - k3*eps
        # ddtheta=deps+dzeta
        # ddtheta_c=(dtheta_c-dtheta_c_last)/dt
        ddtheta_m=0

    Vtd=R(delta-theta)@np.array([nu,0])
    dVtd=R(delta-theta)@np.array([dnu,(ddelta-dtheta)*nu])
    
    b=np.vstack((-dtheta_m*(R(pi/2)@Vt).reshape((2,1)),0)).flatten()
    k=3
    dVt=(dVtd -y1*np.array([sin(theta),cos(theta)])-k*(Vt-Vtd)).reshape((2,1))
    F=A_plus@(np.vstack((*dVt,0)).flatten()-b)
    F=F.flatten()

    return np.array([F[0],F[1],F[2],ds])



#Geometric parameters of the robot
alpha2=pi/2
alpha3=-pi/2
d1=d2=d3=1
d1=1
L=0.5
r=0.25
m=1
J=0.1
D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
A=1/m*np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-r/J,-r/J,-r/J]])@D
A_plus=np.linalg.pinv(A) #penrose inverse




#Drawing and window info
dt,w_size,w_shift= 0.01,15,7
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

path_to_follow=mat_reading() #The path the robot has to follow
x0=np.array([px0,py0,u0,v0,s0,theta0,dtheta0]) # (x,y,vu,vv,theta_m,dtheta_m)
u0=np.array([f1_0,f2_0,f3_0,ds0]) # (f1,f2,f3)
path=[x0[0:2]] #Red dots on the screen, path that the robot follows

key=np.array([0,0,0]) #To control the robot using the keyboard
x=x0
u=u0


draw,end_of_path=1,0

for t in np.arange(0,100,dt):
    if end_of_path:
        print("End of simulation")
        break
    #Drawing and screen update
    X=x[0:2]
    Vt=x[2:4]
    theta_m,dtheta_m=x[5:7]
    F = path_info_update(path_to_follow,x[4])
    theta_c=F.psi
    theta = theta_m-theta_c

    s1,y1 = R(theta_c).T@(X-F.X)
    
    ds=u[-1]
    dtheta_c = F.C_c*ds
    ds1,dy1= R(theta)@Vt-ds*np.array([1-F.C_c*y1,F.C_c*s1])
    dvr_last=state(x,u)[0:2]
    dY_last=np.array([ds1,dy1])
    last_values=[Vt,dY_last,dvr_last,F,u[-1]]


    if draw and (t/dt)%20==0:
        s=x[4]
        ax.clear()
        ax.set_xlim(-w_size-w_shift,w_size-w_shift)
        ax.set_ylim(-w_size,w_size)
        show_info(ax,path_to_follow,X,Vt,theta_m,u,[0,alpha2,alpha3,L,r],[s,t],path,[w_size,w_shift],forces=True,speed=False)     
        draw_crab(X,theta_m,ax,L,r)
        if (t/dt)%30==0: #Change the value to change the frequency of the red dots of the path
            path.append(X)
        plt.pause(10**-10)
    #Update of the state of the simulation
    t_break=t
    T.append(t)
    state_info.append(x)
    x=rungeKutta2(x,u,dt,state)
    u=dynamic_controller2(x,last_values)
    x[4]=max(0,x[4])
    x[5]=sawtooth(x[5])
