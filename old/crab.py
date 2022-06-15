from cmath import atan
from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from numpy import cos,sin,tanh,arctan,pi
import keyboard

def rungeKutta2(x,u,h,f):
    x = x + h*(0.5*f(x,u) + 0.5*f(x + h*f(x,u),u))
    return x

def f(X,u):
    x,y,vu,vv,theta=X
    u1,u2,u3=u
    dx,dy=rot(theta)@(X[2:4])
    dvu=1/(m*R)*u1
    dvv=1/(m*R)*u2
    return np.array([dx,dy,dvu,dvv,-L/(J*R)*u3])

dt,w_size= 0.1,15
x=array([0.5,-7,0,0]) #x,y,v,θ
path=[]
P=np.array([0,0])
R,L=1,1
m,J=1,1

fig, ax = plt.subplots(figsize=(7,7))

def rot(th):
    theta=(th/180)*pi
    return np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
def draw_crab(P,theta):
    n=25
    wheel_size=0.25
    T=np.linspace(0,2*pi,n)
    body=plt.Circle(P,L,edgecolor='#1F6EE1',facecolor='none')
    wheel1 = plt.Rectangle(P+rot(theta)@(L*np.array([0,1])-np.array([R/2,0])), R, wheel_size,angle=theta, linewidth=1,edgecolor='g',facecolor='none')
    wheel2 = plt.Rectangle(P+rot(theta+120)@(L*np.array([0,1])-np.array([R/2,0])), R, wheel_size, linewidth=1,angle=theta+120,edgecolor='r',facecolor='none')
    wheel3 = plt.Rectangle(P+rot(theta-120)@(L*np.array([0,1])-np.array([R/2,0])), R, wheel_size, linewidth=1,angle=theta-120,edgecolor='r',facecolor='none')
    
    
    ax.add_patch(body)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)
    ax.add_patch(wheel3)

def controller(X_crab,Xd,dXd,ddXd):
    X=X_crab[0:2]
    dx,dy=rot(180*X_crab[4]/pi)@(X_crab[2:4])
    dX=np.array([dx,dy])
    alpha0=2
    alpha1=2*alpha0
    v_hat=alpha0*(Xd-X)+alpha1*(dXd-dX)+ddXd
    a1,a2=-L/(J*R)*rot(180*X_crab[4]/pi+90)@X_crab[2:4]
    A=np.array([[1/(m*R)*cos(X_crab[4]),-1/(m*R)*sin(X_crab[4]),a1],[1/(m*R)*sin(X_crab[4]),1/(m*R)*cos(X_crab[4]),a2]])
    pseudo_inv_A = np.linalg.pinv(A)
    return pseudo_inv_A@v_hat


X=np.array([10,-10,0,0,0])
u=np.array([0,0,0])

def traj(t):
    return np.array([10*sin(0.2*t),sin(t)])
    # return 8*(1+a*sin(w2*t))*np.array([cos(w1*t),sin(w1*t)])

w1=0.1
w2=1
a=0.2

for t in arange(0,500,dt):
    #Trajectory
    P=traj(t)
    dP=(traj(t+dt)-traj(t))/dt
    ddP=(traj(t+2*dt)-2*traj(t+dt)+traj(t))/(dt**2)
    # dP=w2*a*sin(w2*t)*np.array([cos(w1*t),sin(w1*t)]) + (1+a*sin(w2*t))*np.array([cos(w1*t),sin(w1*t)])
    # dP=8*dP
    # ddP=(-(w1**2+w2**2)*a*sin(w2*t)-w1**2)*np.array([cos(w1*t),sin(w1*t)]) + w1*w2*a*cos(w2*t)*np.array([-sin(w1*t),cos(w1*t)])
    # ddP=8*ddP
    u=controller(X,np.array([1,1]),0,0)
    
    #Keyboard input for controling the robot
    if keyboard.is_pressed("up"):
        u[0]=1
    elif keyboard.is_pressed("down"):
        u[0]=-1
    if keyboard.is_pressed("left"):
        u[2]=-10
    elif keyboard.is_pressed("right"):
        u[2]=10
    if keyboard.is_pressed("space"):
        break
    
    #Plot and update
    ax.clear()
    ax.set_xlim([-w_size,w_size])
    ax.set_ylim([-w_size,w_size])

    T=np.linspace(0,100,500)
    plt.plot(*(traj(T)))
    plt.scatter(*P)
    ax.text(-17, -18, str(round(u[0],2))+' ' +str(round(u[1],2))+ ' ' +str(round(u[2],2)), style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    draw_crab(X[0:2],180*X[4]/pi-90)
    
    X=rungeKutta2(X,u,dt,f)

    plt.pause(0.0001)


