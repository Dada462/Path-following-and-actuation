import numpy as np
# from numpy import sin,cos,tanh,sign,pi
import matplotlib.pyplot as plt
from sympy import Matrix,sin,cos,Identity
import sympy as sp
from numpy import pi

A=np.array([[1,1,1,1],[10**(-5),0,0,0],[-1,1,-1,1]])
# w,v=np.linalg.eig(np.linalg.pinv(A)@A)
print(np.linalg.pinv(A))

# n=10**2
# t=np.linspace(-pi,pi,n)
# circle=np.array([cos(t),sin(t)])
# v=10
# beta=1.1
# A=np.array([[v**2*cos(beta)**2/(v**2*cos(beta)**2 + sin(beta)**2), v*sin(2*beta)/(-v**2*cos(2*beta) - v**2 + cos(2*beta) - 1)], [v*sin(2*beta)/(-v**2*cos(2*beta) - v**2 + cos(2*beta) - 1), sin(beta)**2/(v**2*cos(beta)**2 + sin(beta)**2)]])
# plt.plot(*(A@circle))
# plt.show()

# def z(beta,v):
#     return 0
# v=1.5
# det=(4*v**6*sin(beta)**2*cos(beta)**6 - v**6*sin(2*beta)**2*cos(beta)**4 - 2*v**4*sin(beta)**2*sin(2*beta)**2*cos(beta)**2 - 8*v**4*sin(beta)**2*cos(beta)**6 + 8*v**4*sin(beta)**2*cos(beta)**4 - v**2*sin(beta)**4*sin(2*beta)**2 + 4*v**2*sin(beta)**2*cos(beta)**6 - 8*v**2*sin(beta)**2*cos(beta)**4 + 4*v**2*sin(beta)**2*cos(beta)**2)/(4*v**8*cos(beta)**8 + 8*v**6*sin(beta)**2*cos(beta)**6 - 8*v**6*cos(beta)**8 + 8*v**6*cos(beta)**6 + 4*v**4*sin(beta)**4*cos(beta)**4 - 16*v**4*sin(beta)**2*cos(beta)**6 + 16*v**4*sin(beta)**2*cos(beta)**4 + 4*v**4*cos(beta)**8 - 8*v**4*cos(beta)**6 + 4*v**4*cos(beta)**4 - 8*v**2*sin(beta)**4*cos(beta)**4 + 8*v**2*sin(beta)**4*cos(beta)**2 + 8*v**2*sin(beta)**2*cos(beta)**6 - 16*v**2*sin(beta)**2*cos(beta)**4 + 8*v**2*sin(beta)**2*cos(beta)**2 + 4*sin(beta)**4*cos(beta)**4 - 8*sin(beta)**4*cos(beta)**2 + 4*sin(beta)**4)
# plt.plot(beta,det)
# plt.show()



beta,alpha2, alpha3, d1, d2, d3, m, J, r, A, u,v,nu=sp.symbols('beta alpha2 alpha3 d1 d2 d3 m J r A u v nu',real=True)

# A=Matrix([[0,d2*sin(alpha2),d3*sin(alpha3)],[-d1,-d2*cos(alpha2),-d3*cos(alpha3)],[-d1*r/J,-d2*r/J,-d3*r/J]])
# A=Matrix([[0,d2*sin(alpha2),d3*sin(alpha3)],[0,-d2*cos(alpha2),-d3*cos(alpha3)],[0,-d2*J/r,-d3*J/r]])

# A=Matrix([[0,cos(beta),-cos(beta)],[0,-sin(beta)/v,sin(beta)/v],[0,-r/J,-r/J]])
# A=Matrix([[1,1,1,1],[beta,0,0,0],[-1,1,-1,1]])
# A=Matrix([[1,1,1,1],[beta,0,0,0],[-1,1,-1,1]])
# A=Matrix([[ 0.7*(1-beta),  1, -1, -1],
#  [ 0.7*(1+beta),  0,  0,  0],
#  [-beta,  0.7, -0.7,  0.7]])

# B=A.pinv()
# B.simplify()
# C=A@B
# C.simplify()
# print(B)
# print(C)
# print(Matrix([[1,0,0],[0,1,0],[0,0,1]])-C)
