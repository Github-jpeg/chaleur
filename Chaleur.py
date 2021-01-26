import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#### Définition des fonctions :

#solution numérique
def equation_de_chaleur(u,t):
    dudt = np.zeros(X.shape)
    for i in range(1, N-1):
        dudt[i] = (((u[i + 1] - 2*u[i] + u[i - 1]) / dx**2))

    return dudt

#### Programme principale :

N = 10
M=1000
L = 1.0
X=np.linspace(0,L,num=N)

dx= L / (N - 1)

t=np.linspace(0.0, L, M)
# conditions initiales 1D
T=np.zeros(len(X))            # la température initiale
T[0]=1                         # température au bord
T[-1]=1                         # température au bord


#### Méthode Euler :
def Euler(F,x0,T):
    n=len(T)
    X=[x0]
    for i in range(n-1):
        y=X[-1]+(T[i+1]-T[i])*F(X[-1],T[i])
        X.append(y)
    return np.array(X)


sol=Euler(equation_de_chaleur,t,T)
ax = Axes3D(plt.figure())
t,X = np.meshgrid(t, X , indexing = 'ij')
ax.plot_surface(t,X, -np.array(solT)+L*np.ones(solT.shape),cmap='plasma')
plt.show()

#### Méthode Odeint :
"""solT = si.odeint(equation_de_chaleur, T, t)

ax = Axes3D(plt.figure())
t,X = np.meshgrid(t, X , indexing = 'ij')
ax.plot_surface(t,X, -np.array(solT)+L*np.ones(solT.shape),cmap='plasma')
plt.show()"""
