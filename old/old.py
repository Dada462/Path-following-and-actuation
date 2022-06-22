# def key_press():
#     global u_kin,end_of_path
#     u_kin_last=u_kin
#     if keyboard.is_pressed("up"):
#         u_kin[0]=3
#     elif keyboard.is_pressed("down"):
#         u_kin[0]=-3
#     else:
#         u_kin[0]=u_kin_last[0]

#     if keyboard.is_pressed("left"):
#         u_kin[1]=3
#     elif keyboard.is_pressed("right"):
#         u_kin[1]=-3
#     else:
#         u_kin[1]=u_kin_last[1]
    
#     if keyboard.is_pressed("a"):
#         u_kin[2]=3
#     elif keyboard.is_pressed("e"):
#         u_kin[2]=-3
#     else:
#         u_kin[2]=u_kin_last[2]
#     if keyboard.is_pressed("space"):
#         end_of_path=1


zeta=ddelta-gamma*y1*u*(sin(theta) - sin(delta)) / sawtooth(theta - delta) - k2*sawtooth(theta-delta)
dzeta=dddelta - gamma*y1*u*((dtheta*cos(theta)-ddelta*cos(delta))*sawtooth(theta-delta)-(dtheta-ddelta)*(sin(theta)-sin(delta)))/(sawtooth(theta-delta)**2)-gamma*(dy1*u+du*y1)*(sin(theta) - sin(delta)) / sawtooth(theta - delta)-k2*(dtheta-ddelta)
eps=dtheta-zeta
deps=-1/gamma*sawtooth(theta-delta) - k3*eps
ddtheta=deps+dzeta
ddtheta_c=(dtheta_c-dtheta_c_last)/dt
ddtheta_m=ddtheta+ddtheta_c
dddelta=-pi/2*(Kdy1*(ddnu*y1+2*dnu*dy1 +nu*ddy1)*(1-tanh(Kdy1*nu*y1)**2) -2*Kdy1**2*(dnu*y1+nu*dy1)**2*(1-tanh(Kdy1*nu*y1)**2)*tanh(Kdy1*nu*y1))
