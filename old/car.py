from cmath import atan
from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from numpy import cos,sin,tanh,arctan,arctan2
import keyboard


def f_dynamic(x,u):
    return np.array([x[2]*cos(x[3]),x[2]*sin(x[3]),u[0],u[1]])

def f_kinematic(x,u):
    return np.array([u[0]*cos(x[3]),u[0]*sin(x[3]),0,u[1]])

theta_a=1
k_delta=1
def delta(y1,v):
    return (-theta_a*np.tanh(k_delta*y1*v))%(2*pi)

def controller(X,P,dP,t):
    global dV,s1,y1,ds
    x,y,_,theta_m=X
    
    #########U,V,dV
    U=dP/np.linalg.norm(dP.reshape((2,1)))
    V=np.array([-U[1],U[0]])
    dV=w*(1/2*sin(2*w*t)*(1+cos(w*t)**2)**(-0.5)*V + 1/sqrt(1+cos(w*t)**2)*np.array([sin(w*t),0]))
    #########
    theta_c=np.arctan2(U[1],U[0])
    theta=sawtooth(theta_m-theta_c)
    k1=1
    k2=1
    gamma=1

    s1=np.dot((X[0:2]-P),U)
    y1=np.dot((X[0:2]-P),V)

    dy1=np.dot(np.array([X[2]*cos(X[3]),X[2]*sin(X[3])])-ds*dt*U,V) + np.dot(X[0:2]-P,dV)
    ddelta=-theta_a*(1-tanh(k_delta*y1*v)**2)*dy1
    ds=v*cos(theta)+k1*s1
    dtheta=ddelta-gamma*y1*v*(sin(theta)-sin(delta(y1,v)))/sawtooth(theta-delta(y1,v))-k2*sawtooth(theta-delta(y1,v))
    return  ds,dtheta,U,V

v=1.5
dt,w_size= 0.1,20
fig,ax=plt.subplots(figsize=(8,7))
x0=np.array([-2,7,v,pi/2]) #x,y,v,theta
x=x0
path=[]
t_break=0
T,Y=[],[]
w=0.5
ds=1
P=np.array([0,0])
dP=w*np.array([1,1])
draw=1
for t in arange(0,50,dt):
    
    ds,dtheta,U,V=controller(x,P,dP,t)
    dP=w*np.array([1,cos(w*t)])
    U=dP/np.linalg.norm(dP)
    P=P+ dt*ds*U
    u=np.array([v,dtheta])

    #Drawing
    if draw:
        ax.clear()
        l=np.linspace(-5,100,100)
        ax.plot(w*l,sin(w*l))
        ax.scatter(*P)
        ax.quiver(*P,*dV)
        ax.quiver(*P,*U,color='red',width=0.005)
        ax.quiver(*P,*V,color='red',width=0.005)
        ax.set_xlim(-w_size+7,w_size+7)
        ax.set_ylim(-w_size,w_size)
        ax.text(0, w_size +1, 's1='+str(round(s1,2)) + '\n' +'y1='+str(round(y1,2)), style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        ax.plot([P[0],P[0]+s1*U[0]],[P[1],P[1]+s1*U[1]])
        ax.plot([P[0],P[0]+y1*V[0]],[P[1],P[1]+y1*V[1]])
        draw_tank(x[[0,1,3]].reshape((3,1)),'red',0.15) # x,y,θ

        if (t/dt)%5==0:
            path.append(x[0:2])
        for point in path:
            ax.scatter(*point,c='red',s=10)
        if keyboard.is_pressed("space"):
            break 
    T.append(t)
    Y.append(x)
    #Update and drawings    
    x=x+dt*f_kinematic(x,u)
    t_break=t
    if draw:
        pause(10e-10)

ax.clear()
Y=np.array(Y)
ax.plot(T,Y[:,3],color='blue')
ax.set_xlabel('temps')
ax.set_ylabel('θm(t) et θc(t)')
ax.text(t_break-t_break/3, 1.1, r'$θc(t)$',color='green')
ax.text(t_break-t_break/3, 1, r'$θm(t)$',color='blue')
ax.plot(w*np.arange(0,t_break,dt),sin(w*np.arange(0,t_break,dt)),color='green')
pause(0)