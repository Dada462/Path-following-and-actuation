import numpy as np
from numpy import reshape, sin,cos,tanh,sign,pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def R(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def state(t,X,u1,u2):
    x,y,psi=X
    Vt=np.array([u1,0])
    V=R(psi)@Vt
    dpsi=u2
    dX=np.array([V[0],V[1],dpsi])
    return dX
n=4000

X,psi,s,C_c= np.array([0,0]),0,0,0
Chemin=n*[[X,psi,s,C_c]]
dt = 0.01
for i in range(1,n):
    t = i*dt
    Vbtarget = 1
    Wbtarget = cos(0.1*t)*sin(0.2*t)
    C = np.array([Vbtarget,Wbtarget])
    X=Chemin[i-1][0]
    psi=Chemin[i-1][1]
    xt_last = np.array([X[0],X[1],psi])
    sol=solve_ivp(state,[0,dt],xt_last,args=C)
    x45=sol.y

    Chemin[i][0] = x45[-1,0:2]
    Chemin[i][1] = x45[-1,2]
    blabla=(Chemin[i][0] - Chemin[i-1][0])
    ds=blabla[0]**2+blabla[1]**2
    ds=ds**0.5
    print(i,Chemin[i][1],ds)
    Chemin[i][2] = Chemin[i-1][2] + ds;
    Chemin[i][3] = (Chemin[i][1] - Chemin[i-1][1])/ds;

for x in Chemin:
    X,psi,s,C_c=x
    plt.plot(*X)
plt.show()