from cmath import atan
from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from numpy import cos,sin,tanh,arctan,arctan2
import keyboard
import scipy.io

class lievre:
    def __init__(self,X,psi,s,C_c):
        self.s = s
        self.psi = psi
        self.C_c = C_c
        self.X = X
    def __str__(self):
        return str(self.s) + ' ' + str(self.psi) + ' ' + str(self.C_c) + ' ' + str(self.X)

def mat_reading():
    mat = scipy.io.loadmat('PATH.mat')
    
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
    Chemin=lievre(X,psi,s,C_c)
    return Chemin
Chemin=mat_reading()

def find(elements,s):
    inds = [i for (i, val) in enumerate(elements) if val >s]
    if inds!=[]:
        return inds[0]
    else:
        return "None"

def Interrogation_chemin(Chemin,s):
    I=find((Chemin.s),s)
    Lievre=lievre(0,0,0,0)
    if I=='None':
        print('fin du chemin')
        return None
    else:
        Delta_S_Lievre = (Chemin.s)[I]-(Chemin.s)[I-1]
        ratio_S_Lievre = (s-Chemin.s[I-1])/Delta_S_Lievre
        Lievre.s = s
        Lievre.psi = Chemin.psi[I-1]*(1-ratio_S_Lievre) + Chemin.psi[I]*(ratio_S_Lievre)
        Lievre.C_c = Chemin.C_c[I-1]*(1-ratio_S_Lievre) + Chemin.C_c[I]*(ratio_S_Lievre)
        Lievre.X = Chemin.X[I-1]*(1-ratio_S_Lievre) + Chemin.X[I]*(ratio_S_Lievre)
        return Lievre

def f_kinematic(x,u):
    psi=x[2]
    M_C=np.array([[cos(psi),0],[sin(psi),0],[0,1]])
    V_B = u[0];
    W_B = u[1];
    dx = M_C@np.array([V_B,W_B])

    return dx

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

dt,w_size= 0.1,10
fig,ax=plt.subplots(figsize=(8,7))
T,Y=[],[]
S=[]
path=[]
t_break=0
draw=0

X = np.array([-2,1])
psi = pi;
V_B = 1;
W_B = 0;
x=np.hstack((X[0],X[1],psi)).flatten()
psilievre=0
s=0
for t in arange(0,500,dt):
    i=int(t/dt)
    M_C=np.array([[cos(psi),0],[sin(psi),0],[0,1]])
    V_0 = (M_C@np.array([V_B,W_B]))[0:2]
    
    #Chemin
    Lievre = Interrogation_chemin(Chemin,s)
    if Lievre==None:
        break
    X_e = Lievre.X -X
    Theta = Lievre.psi - psi
    R_S_0 = np.array([[cos(Lievre.psi),-sin(Lievre.psi)],[sin(Lievre.psi),cos(Lievre.psi)]])
    
    X_e_F = (R_S_0.T)@X_e
    s1 = X_e_F[0]
    y1 = X_e_F[1]
    
    K1=10
    
    dot_s = V_B*cos(Theta)  - K1*s1
    
    dot_X_LIEVRE_F =  np.array([dot_s,0])
    
    dot_X_e_F = dot_X_LIEVRE_F - (R_S_0.T)@V_0
    
    dot_s1 = dot_X_e_F[0]
    dot_y1 = dot_X_e_F[1]
    
    dot_psi_LIEVRE = Lievre.C_c*dot_s
    
    Kdy1=10
    Delta = -np.arctan(Kdy1*y1)
    Delta_Prime = -Kdy1/(1+(Kdy1*y1)**2)
    dot_Delta = Delta_Prime*dot_y1

    K=10
    dot_Theta_Control = dot_Delta + K*(Delta - Theta)
    
    dot_psi = dot_psi_LIEVRE - dot_Theta_Control
    W_B = dot_psi
    
    u = np.array([V_B,W_B])
    # print(dot_s)
    s = s +dot_s*dt
    
    if s<0:
        s=0
    #Drawing
    if draw:
        ax.clear()
        ax.set_xlim(-w_size+7,w_size+7)
        ax.set_ylim(-w_size,w_size)
        # ax.text(0, w_size +1, 's1='+str(round(s1,2)) + '\n' +'y1='+str(round(y1,2)), style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        draw_tank(x[[0,1,2]].reshape((3,1)),'red',0.15) # x,y,θ
        ax.plot(*Chemin.X.T)
        if (t/dt)%5==0:
            path.append(x[0:2])
        for point in path:
            ax.scatter(*point,c='red',s=10)
        if keyboard.is_pressed("space"):
            break
    #Update and drawings 
    psilievre=psilievre+dot_psi_LIEVRE*dt
    x=x+dt*f_kinematic(x,u)
    X=x[0:2]
    psi=x[2]
    t_break=t
    T.append(t)
    Y.append(x)
    S.append(psilievre)
    if draw:
        pause(10e-10)

ax.clear()
Y=np.array(Y)
S=np.array(S)
ax.plot(T,Y[:,2]-S,color='blue')
ax.set_xlabel('temps')
ax.set_ylabel('θm(t)')
ax.text(t_break-t_break/3, 1, r'$θm(t)$',color='blue')
plt.show()