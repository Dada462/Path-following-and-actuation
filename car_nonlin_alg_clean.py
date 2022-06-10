from cmath import atan
from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from numpy import cos,sin,tanh,arctan,arctan2
import keyboard
import scipy.io

class path_data:
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
    M_C=np.array([[cos(psi),0],[sin(psi),0],[0,1]])
    V_B = u[0];
    W_B = u[1];
    dx = M_C@np.array([V_B,W_B])
    return dx

def controller(x,u_last,path_to_follow):
    global end_of_path
    X=x[0:2]
    psi=x[2]
    
    M_C=np.array([[cos(psi),0],[sin(psi),0],[0,1]])
    V_0 = (M_C@u_last)[0:2]

    F = path_info_update(path_to_follow,s)
    if F==None:
        end_of_path=1
        return 0,0
    X_e = F.X -X
    theta = F.psi - psi
    R_S_0 = np.array([[cos(F.psi),-sin(F.psi)],[sin(F.psi),cos(F.psi)]])
    
    X_e_F = (R_S_0.T)@X_e
    s1,y1 = X_e_F
    K1=10
    ds = v*cos(theta)-K1*s1
    
    #Frenet
    dX_F =  np.array([ds,0])
    ds1,dy1=dX_F - (R_S_0.T)@V_0
    dpsi_F = F.C_c*ds
    #delta
    Kdy1=10
    delta = -np.arctan(Kdy1*y1)
    ddelta = -Kdy1/(1+(Kdy1*y1)**2)*dy1

    K=10
    dpsi = dpsi_F - (ddelta + K*(delta - theta))
    
    u = np.array([v,dpsi])
    return u,ds

dt,w_size= 0.1,10
fig,ax=plt.subplots(figsize=(8,7))
T,state_info,path=[],[],[]
t_break=0

v=1
x0=np.array([-5,-10,0])
u0=np.array([v,0])

x=x0
u=u0
s=0
draw,end_of_path=1,0

for t in arange(0,500,dt):
    
    u,ds=controller(x,u,path_to_follow)
    # u[0]=1.5+sin(t)
    if end_of_path:
        print("end of path")
        break
    if s<0:
        s=0
    #Drawing
    if draw:
        ax.clear()
        ax.set_xlim(-w_size+4,w_size+4)
        ax.set_ylim(-w_size,w_size)
        # ax.text(0, w_size +1, 's1='+str(round(s1,2)) + '\n' +'y1='+str(round(y1,2)), style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        draw_tank(x[[0,1,2]].reshape((3,1)),'#F23F37',0.15) # x,y,θ
        ax.scatter(*path_info_update(path_to_follow,s).X,c='#34F44C')
        ax.plot(*path_to_follow.X.T,c='#3486F4')
        if (t/dt)%5==0:
            path.append(x[0:2])
        for point in path:
            ax.scatter(*point,c='red',s=10)
        if keyboard.is_pressed("space"):
            break
    #Update and drawings
    x=x+dt*state(x,u)
    s=s+ds*dt
    t_break=t
    T.append(t)
    state_info.append(x)
    if draw:
        pause(10e-10)

#Plotting results
ax.clear()
state_info=np.array(state_info)
ax.plot(T,state_info[:,2])
ax.set_xlabel('temps')
ax.set_ylabel('θm(t)')
ax.text(t_break-t_break/3, 1, r'$θm(t)$',color='#3486F4')
plt.show()