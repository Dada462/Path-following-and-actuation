from cmath import atan
from numpy import cos,sin,tanh,arctan,pi
import scipy.io
import numpy as np
import keyboard
import matplotlib.pyplot as plt


dt,w_size,w_shift= 0.01,6,0
fig,ax=plt.subplots(figsize=(8,7))
key=np.array([0,0,0])
u=key

#To control the robot with the keyboard
def key_press():
    global key,u
    alpha=5
    if keyboard.is_pressed("up"):
        key[1]=alpha
    elif keyboard.is_pressed("down"):
        key[1]=-alpha
    else:
        key[1]=0
    if keyboard.is_pressed("left"):
        key[0]=-alpha
    elif keyboard.is_pressed("right"):
        key[0]=alpha
    else:
        key[0]=0
    
    if keyboard.is_pressed("e"):
        key[2]=alpha
    elif keyboard.is_pressed("d"):
        key[2]=-alpha
    else:
        key[2]=0
    if keyboard.is_pressed("space"):
        end_of_path=1
    u=key

def R(theta):
    return np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])

a,b=1,1
def draw_cube(P,theta):
    theta_deg=180*theta/pi-90
    
    #The rectangular body
    body=plt.Rectangle(P-R(theta)@np.array([a/2,-b/2]), a, b,angle=theta_deg, linewidth=1,edgecolor='g',facecolor='none')
    
    #Angle of the forces
    alpha1=pi/3
    alpha2=pi/3
    alpha3=pi/3
    alpha4=pi/3

    #Plotting of the forces
    ax.quiver(*(P+R(theta)@np.array([a/2,b/2])),*R(theta+alpha1)[:,0],color='red',scale=10)
    ax.quiver(*(P+R(theta)@np.array([a/2,-b/2])),*R(theta+alpha2)[:,0],color='red',scale=10)
    ax.quiver(*(P+R(theta)@np.array([-a/2,-b/2])),*R(theta+alpha3)[:,0],color='red',scale=10)
    ax.quiver(*(P+R(theta)@np.array([-a/2,b/2])),*R(theta+alpha4)[:,0],color='red',scale=10)
    
    ax.add_patch(body)
P=np.array([0,0])
theta=0

for t in np.arange(0,30,dt):
    ax.clear()
    ax.set_xlim(-w_size-w_shift,w_size-w_shift)
    ax.set_ylim(-w_size,w_size)
    key_press()
    draw_cube(P,theta)
    
    #To plot foces
    P=P+R(theta)@u[0:2]*dt
    theta=theta+dt*u[2]
    plt.pause(10**-10)