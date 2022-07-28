import numpy as np
from numpy import cos,sin,tanh,arctan,arctan2,pi,arctanh
from tools import rungeKutta2,sawtooth,mat_reading,R,path_info_update,draw_crab,show_info,key_press,R3
from A_mat2 import A
import matplotlib.pyplot as plt

# s=open('state_info_file copy.txt', 'r').read()
# x=s.split('\n')
# f=open('test.txt','w')
# f.write('A=np.array([')
# for var in x:
#     f.write('[')
#     f.write(var)
#     f.write('],')
# f.write('])')
# f.close()

# for var in x:
#     a=[]
#     b=var.split(',')
#     for n in var:
#         a.append(float(n))
#     X.append(a)

n=np.shape(A)[0]
T=np.linspace(0,1,n)
theta_m=(A[:,5])
dtheta_m=(A[:,6])
dt,w_size,w_shift= 0.01,3,0
# beta=arctan2(A[:,3],A[:,2])
# plt.plot(theta_m,dtheta_m)
fig,ax=plt.subplots(figsize=(8,7))
for i in range(len(theta_m)):
    if i%5==0:
        ax.clear()
        ax.set_xlim(-w_size-w_shift,w_size-w_shift)
        ax.set_ylim(-w_size,w_size)
        ax.plot(theta_m[0:i+1],dtheta_m[0:i+1])
        plt.pause(0.001)
# plt.plot(T,sawtooth(20*(T+0.05)))
# plt.show()


# #Drawing and window info
# dt,w_size,w_shift= 0.01,15,-7
# fig,ax=plt.subplots(figsize=(8,7))

# #Geometric parameters of the robot
# alpha2=pi/2
# alpha3=-pi/2
# d1=d2=d3=1
# d1=0
# L=1
# r=0.25
# m=1
# J=1
# D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
# path_to_follow=mat_reading() #The path the robot has to follow
# n=np.shape(A)[0]
# i=0
# for t in np.linspace(0,1,n):
#     if i%15==0:
#         x=A[int(t*n)]
#         X=x[0:2]
#         Vt=x[2:4]
#         theta_m,dtheta_m=x[5:7]
#         s=x[4]
        # ax.clear()
        # ax.set_xlim(-w_size-w_shift,w_size-w_shift)
        # ax.set_ylim(-w_size,w_size)
#         show_info(ax,path_to_follow,X,Vt,theta_m,[0,0,0,0],[0,alpha2,alpha3,0.5,r],[s,t],[],[w_size,w_shift],forces=False,speed=False)     
#         draw_crab(X,theta_m,ax,0.5,r)
#         plt.pause(0.01)
#     i+=1





