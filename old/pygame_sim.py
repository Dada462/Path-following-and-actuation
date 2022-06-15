import pygame
import numpy as np
from numpy import sin,cos,pi
from cmath import atan
from numpy import cos,sin,tanh,arctan,pi
import matplotlib.pyplot as plt

WIN=pygame.display.set_mode((800,800))
pygame.display.set_caption("Simulation")
pygame.init()
clock = pygame.time.Clock()
font = pygame.font.Font('freesansbold.ttf', 25)

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
x=np.array([0.5,-7,0,0]) #x,y,v,θ
path=[]
P=np.array([0,0])
R,L=1,1
m,J=1,1


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

scale=25
wheel_size=0.25
R*scale, wheel_size*scale
rotation=0
image_orig = pygame.Surface((R*scale, wheel_size*scale))
image_orig.set_colorkey((0,0,0))
image_orig.fill((0,255,0))
image = image_orig.copy()  
image.set_colorkey((0,0,0))

rect = image.get_rect()
rect.center = (400,400)
for t in np.arange(0,500,dt):
    clock.tick(200) # pour limiter la vitesse de la boucle while à FPS boucles/sec
    WIN.fill((220, 228, 231))
    l,L=0,0
    # rect_test = pygame.Rect((0, 0), (100,50))
    # rect_test.center = 400+100*cos(0.1*t),500+100*sin(0.1*t)
    # pygame.draw.rect(WIN, (255, 255, 255), rect_test)
    # text = font.render('test', True, (0,0,0))
    # A=text.get_rect()
    # A.center=rect_test.center
    # WIN.blit(text,A)
    
    #Trajectory
    P=traj(t)
    dP=(traj(t+dt)-traj(t))/dt
    ddP=(traj(t+2*dt)-2*traj(t+dt)+traj(t))/(dt**2)
    u=controller(X,P,dP,ddP)

    
    theta=X[4]

    p1=(P+rot(theta)@(L*np.array([0,1])-np.array([R/2,0])))+400
    p2=(P+rot(theta+120)@(L*np.array([0,1])-np.array([R/2,0])))+400
    p3=(P+rot(theta-120)@(L*np.array([0,1])-np.array([R/2,0])))+400

    wheel1 = pygame.Rect(p1, (R*scale, wheel_size*scale))
    wheel2 = pygame.Rect(p2, (R*scale, wheel_size*scale))
    wheel3 = pygame.Rect(p3, (R*scale, wheel_size*scale))
    
    pygame.draw.circle(WIN, '#1F6EE1', (X[0]+400,X[1]+400), scale)
    
    
    rotation = X[4]+120
    new_image = pygame.transform.rotate(image_orig , rotation)
    rect = new_image.get_rect()
    rect.center = p1
    WIN.blit(new_image , rect)

    pygame.draw.rect(WIN, (255, 255, 255), wheel1)
    pygame.draw.rect(WIN, (255, 255, 255), wheel2)
    pygame.draw.rect(WIN, (255, 255, 255), wheel3)
    
    for event in pygame.event.get(): #détection des ordres du joueur
        if event.type==pygame.QUIT: #pour quitter
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE: #Donne le score de l'échiquier
                print('space')
            if event.key == pygame.K_UP: #Donne le score de l'échiquier
                u[0]=1
            if event.key == pygame.K_DOWN: #Donne le score de l'échiquier
                u[0]=-1
            if event.key == pygame.K_RIGHT: #Donne le score de l'échiquier
                u[2]=10
            if event.key == pygame.K_LEFT: #Donne le score de l'échiquier
                u[2]=-10
    pygame.display.update()

