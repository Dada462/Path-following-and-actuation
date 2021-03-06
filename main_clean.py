from cmath import atan
from numpy import cos,sin,tanh,arctan,arctan2,pi
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from tools import rungeKutta2,sawtooth,mat_reading,R,path_info_update,draw_crab,show_info,key_press,R3
import joystick


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
    if nu<0.2:
        return np.array([1,-1,1,0])
    dnu=dVt.T@Vt/nu
    F = path_info_update(path_to_follow,s)
    if F==None:
        end_of_path=1
        return np.array([0,0,0,0])
    theta_c=F.psi
    theta = sawtooth(theta_m-theta_c)
    beta=arctan2(Vt[1],Vt[0])
    psi=sawtooth(theta+beta)

    s1,y1 = R(theta_c).T@(X-F.X)
    
    ks=1
    ds=cos(psi)*nu +ks*s1
    dtheta_c = F.C_c*ds
    ds1,dy1= R(psi)@np.array([nu,0])-ds*np.array([1-F.C_c*y1,F.C_c*s1])
    dtheta=dtheta_m-dtheta_c


    Kdy1=2
    delta = -pi/2*tanh(Kdy1*y1*nu)
    ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*nu)**2)*(dy1*nu+dnu*dy1)
    
    k2=2
    dpsi=ddelta - k2*sawtooth(psi-delta)
    vtd=(1-nu)
    dbeta=dpsi-dtheta
    ddtheta_m=-dtheta_m+(pi/4-theta_m)
    W=np.array([vtd,dbeta,ddtheta_m])

    B=np.array([[1,0,0],[0,1/nu,0],[0,0,1]])@R3(beta).T@A
    F=np.linalg.pinv(B)@(W+np.array([0,dtheta_m,0]))

    return np.array([F[0],F[1],F[2],ds])



#Geometric parameters of the robot
alpha2=2*pi/3
alpha3=-2*pi/3
d1=d2=d3=1
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
px0,py0=15,8
u0,v0=0.1,0
theta0,dtheta0=0,0
s0=30

f1_0,f2_0,f3_0=0,0,0
ds0=0

path_to_follow=mat_reading() #The path the robot has to follow
x0=np.array([px0,py0,u0,v0,s0,theta0,dtheta0]) # (x,y,vu,vv,theta_m,dtheta_m)
u0=np.array([f1_0,f2_0,f3_0,ds0]) # (f1,f2,f3)
path=[x0[0:2]] #Red dots on the screen, path that the robot follows
# ax.plot(path_to_follow.s,path_to_follow.C_c)
# ax.plot([36,36],[0.5,-0.5])
# plt.show()

key=np.array([0,0,0]) #To control the robot using the keyboard
x=x0
u=u0

draw,end_of_path=1,0
data=[]

# joy = joystick.XboxController()
# state_info_file=open('state_info_file.txt', 'w')

for t in np.arange(0,10000,dt):
    if end_of_path:
        print("End of simulation")
        break
    #Drawing and screen update
    X=x[0:2]
    Vt=x[2:4]
    theta_m,dtheta_m=x[5:7]

    if draw and (t/dt)%15==0:
        s=x[4]
        ax.clear()
        ax.set_xlim(-w_size-w_shift,w_size-w_shift)
        ax.set_ylim(-w_size,w_size)
        show_info(ax,path_to_follow,X,Vt,theta_m,u,[0,alpha2,alpha3,0.5,r],[s,t],path,[w_size,w_shift],forces=True,speed=False)     
        draw_crab(X,theta_m,ax,0.5,r)
        # u1,v1=Vt
        # nu=round((u1**2+v1**2)**0.5,2)
        ax.text(-w_size/2, w_size+0.5, 's:'+str(round(s,2)), style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
        if (t/dt)%30==0: #Change the value to change the frequency of the red dots of the path
            path.append(X)
        plt.pause(10**-10)
    
    #Update of the state of the simulation
    t_break=t
    T.append(t)
    dVt=state(x,u)[2:4]
    state_info.append(x)
    u=dynamic_controller2(x,dVt)
    # u[0:3]=10*tanh(0.2*u[0:3])
    # turn_left,turn_right,forward,backwards=joy.read()
    # k2=0.4
    # k1=0.2
    # u[1:3]=np.array([k1*(forward-backwards)+k2*turn_left,-k1*(forward-backwards)+k2*turn_left])
    x=rungeKutta2(x,u,dt,state)
    # for var in x:
    #     state_info_file.write(str(var))
    #     if var !=x[-1]:
    #         state_info_file.write(',')
    # state_info_file.write('\n')
    x[4]=max(0,x[4])
    x[5]=sawtooth(x[5])
# state_info_file.close()

ax.clear()
state_info=np.array(state_info)
u=state_info[:,2]
v=state_info[:,3]
ax.plot(T,(u**2+v**2)**0.5, color='red')
ax.set_xlabel('time (s)')
ax.set_ylabel('data')
plt.show()