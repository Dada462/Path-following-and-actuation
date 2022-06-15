

def kin_controller(x,path_to_follow):
    global end_of_path
    Vt=x[2:4]
    vu,vv=Vt[0:2]

    X,theta_m=x[0:2],x[4]
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
    ds1,dy1= (M[0:2,0:2]@R.T)@(Vt[0:2])-dX_F
    dpsi_F = F.C_c*ds

    #delta
    nu=(Vt[0]**2+Vt[1]**2)**0.5
    dnu=1
    Kdy1=1
    delta = -pi/2*tanh(Kdy1*y1*nu)
    ddelta = -pi/2*Kdy1*(1-tanh(Kdy1*y1*nu)**2)*dy1*dnu
    
    K=4
    gamma=0.5
    # y0=0.5
    # vv=-2*tanh((theta-delta))**2*cos(theta)*tanh(y1/y0)
    if theta-delta!=0:
        dpsi = dpsi_F + (ddelta -K*(theta - delta) -gamma*y1*nu*(sin(theta)-sin(delta))/sawtooth(theta-delta))
    else:
        dpsi = dpsi_F + (ddelta -K*(theta - delta) -gamma*y1*(vu +vv))
    u = np.array([vu,vv,dpsi])
    return u,ds


def dynamic_controller(x,u_kin):
    Vt=x[2:4]
    dtheta_m=x[5]
    D=np.array([[d1,0,0],[0,d2,0],[0,0,d3]])
    A=1/m*np.array([[0,sin(alpha2),sin(alpha3)],[-1,-cos(alpha2),-cos(alpha3)],[-r/J,-r/J,-r/J]])@D
    b=np.vstack((-dtheta_m*(R(pi/2)@Vt).reshape((2,1)),0)).flatten()
    A_plus=np.linalg.pinv(A) #penrose inverse
    k=0*(np.eye(3)-A_plus@A)@D@np.array([1,1,1])
    u=A_plus@(u_kin-b) + k
    return u