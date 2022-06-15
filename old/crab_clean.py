from cmath import atan
from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from numpy import cos,sin,tanh,arctan,pi
import keyboard
import scipy.io


def rungeKutta2(x,u,h,f):
    x = x + h*(0.5*f(x,u) + 0.5*f(x + h*f(x,u),u))
    return x

class path_data:
    def __init__(self,X,psi,s,C_c):
        self.s = s
        self.psi = psi
        self.C_c = C_c
        self.X = X
    def __str__(self):
        return str(self.s) + ' ' + str(self.psi) + ' ' + str(self.C_c) + ' ' + str(self.X)

def mat_reading(path='PATH.mat'):
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

def state(x,u):
    psi=x[2]
    M_C=np.array([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]])
    _vu = u[0];
    _vv = u[1];
    w = u[2];
    dx = M_C@np.array([_vu,_vv,w])
    return dx

def rot(th):
    theta=(th/180)*pi
    return np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
L,R=0.5,0.25
def draw_crab(P,theta):
    # n=25
    wheel_size=0.5
    # T=np.linspace(0,2*pi,n)
    body=plt.Circle(P,L,edgecolor='#1F6EE1',facecolor='none')
    wheel1 = plt.Rectangle(P+rot(theta)@(L*np.array([0,1])-np.array([R/2,0])), R, wheel_size,angle=theta, linewidth=1,edgecolor='g',facecolor='none')
    wheel2 = plt.Rectangle(P+rot(theta+120)@(L*np.array([0,1])-np.array([R/2,0])), R, wheel_size, linewidth=1,angle=theta+120,edgecolor='r',facecolor='none')
    wheel3 = plt.Rectangle(P+rot(theta-120)@(L*np.array([0,1])-np.array([R/2,0])), R, wheel_size, linewidth=1,angle=theta-120,edgecolor='r',facecolor='none')
    
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

def controller(x,u_last,path_to_follow):
    global end_of_path
    X=x[0:2]
    psi=x[2]
    
    M_C=np.array([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]])
    V_0 = (M_C@u_last)[0:2]

    F = path_info_update(path_to_follow,s)
    if F==None:
        end_of_path=1
        return np.array([0,0,0]),0
    theta = F.psi - psi
    R_S_0 = np.array([[cos(F.psi),-sin(F.psi)],[sin(F.psi),cos(F.psi)]])
    s1,y1 = (R_S_0.T)@(F.X -X)

    K1=10
    ds = vv*cos(theta)-sat(t)*sin(theta)*vu-K1*s1
    
    #Frenet
    dX_F =  np.array([ds,0])
    ds1,dy1=dX_F - (R_S_0.T)@V_0
    dpsi_F = F.C_c*ds

    #delta
    Kdy1=5
    delta = -pi/2*np.tanh(Kdy1*y1*u_last[0])
    ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*V_0[0])**2)*dy1*u_last[0]
    delta1=delta+pi/2

    K=5
    gamma=5
    dpsi = dpsi_F - (ddelta -K*(theta - delta) -gamma*y1*(vu*(sin(theta)-sin(delta))/(theta-delta) +vv*sat(t)*(cos(theta)-cos(delta1))/(theta-delta) ) )
    
    u = np.array([vu,sat(t)*vv,dpsi])
    return u,ds

def controller2(x,wd,theta_d):
    return np.hstack((3*np.array([[cos(x[2]),-sin(x[2])],[sin(x[2]),cos(x[2])]]).T@(wd-x[0:2]),3*sawtooth(theta_d-x[2])))

def controller3(x,u_last,path_to_follow):
    global end_of_path
    vu,vv=u_last[0:2]

    X,theta_m=x[0:2],x[2]
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
    ds1,dy1= (M[0:2,0:2]@R.T)@(u_last[0:2])-dX_F
    dpsi_F = F.C_c*ds

    #delta
    Kdy1=1
    delta = -pi/2*tanh(Kdy1*y1*u_last[0])
    ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*u_last[0])**2)*dy1*u_last[0]
    delta1=-pi/2*tanh(y1*u_last[1])+pi/2
    
    K=4
    gamma=0.5
    y0=0.5
    vv=-2*sat(t)*tanh((theta-delta))**2*cos(theta)*tanh(y1/y0)
    dpsi = dpsi_F + (ddelta -K*(theta - delta) -gamma*y1*(vu*(sin(theta)-sin(delta))/(theta-delta) +vv*(cos(theta)-cos(delta1))/(theta-delta)))
    u = np.array([vu,sat(t)*vv,dpsi])
    return u,ds

def key_press():
    global u,end_of_path
    if keyboard.is_pressed("up"):
        u[0]=3
    elif keyboard.is_pressed("down"):
        u[0]=-3
    if keyboard.is_pressed("left"):
        u[2]=3
    elif keyboard.is_pressed("right"):
        u[2]=-3
    if keyboard.is_pressed("space"):
        end_of_path=1

dt,w_size,w_shift= 0.01,15,-7
fig,ax=plt.subplots(figsize=(8,7))
T,state_info,data=[],[],[]
t_break=0
vu,vv=3,0
x0=np.array([15,12,pi/2])
u0=np.array([vu,vv,pi/2])

path=[x0[0:2]]
x=x0
u=u0
s=0
draw,end_of_path=1,0

for t in arange(0,500,dt):
    u,ds=controller3(x,u,path_to_follow)
    F = path_info_update(path_to_follow,s)
    if end_of_path:
        print("End of simulation")
        break
    if s<0:
        s=0
    #Keyboard input for controling the robot
    key_press()
    if draw and (t/dt)%5==0:
        ax.clear()
        if t>t0:
            ax.text(w_size, w_size+0.5,'MOTOR DEAD', style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
        ax.set_xlim(-w_size-w_shift,w_size-w_shift)
        ax.set_ylim(-w_size,w_size)
        draw_crab(x[0:2],180*x[2]/pi-90)
        ax.scatter(*path_info_update(path_to_follow,s).X,c='#34F44C')
        ax.plot(*path_to_follow.X.T,c='#3486F4')
        if (t/dt)%10==0:
            path.append(x[0:2])
        for point in path:
            ax.scatter(*point,c='red',s=10)
        info='u1 : '+str(round(u[0],2))+'\n'+'u2 : '+str(round(u[1],2))+ '\n' 'u3 : '+str(round(u[2],2)) + '\n''θ :'+str(round(180*(x[2]-F.psi)/pi,2))+ ' deg'
        ax.text(w_size/2, w_size+0.5,info, style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
        ax.text(0, w_size+1, 'time :' + str(round(t,2)) +' s', style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})    
        plt.pause(10**-10)
    if F!=None:
        data.append(180*(x[2]-F.psi)/pi)
    T.append(t)
    state_info.append(x)
    x=x+dt*state(x,u)
    s=s+ds*dt
    t_break=t

#Plotting results
ax.clear()
ax.plot(T,data)
ax.plot([t0]*2,np.linspace(min(data),max(data),2))
ax.set_xlabel('time (s)')
ax.set_ylabel('data')
plt.show()
