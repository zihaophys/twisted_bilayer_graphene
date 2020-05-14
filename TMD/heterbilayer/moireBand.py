'''
Author: Zihao Wang
This is moire bands of WSe2 on MoSe2
continuum model using plane wave expansion
'''

import numpy as np 
import matplotlib.pyplot as plt 
from numpy import pi, sin, cos, exp, sqrt 
#define constant
theta = 2.0     #degree
V     = 6.6     #meV
psi   = -94.0   #degree
meff  = 0.35    #electron mass
a0    = 3.297   #angstrom
N     = 8       #truncate range
I     = complex(0, 1)

#tune parameter
def Rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], \
        [np.sin(theta), np.cos(theta)]])

theta = theta * pi / 180.0
V = V * 10**(-3)
psi = psi * pi / 180.0
hbar = 4.135667 * 10**(-15)/2/pi
me = 0.51099895 * 10**(6)
m_eff = meff*me
kin = -hbar**2/2/m_eff *9*10**(16)
a0 = a0 * 10**(-10)
a1 = a0 * np.array([1, 0])
a2 = Rot(pi/3).dot(a1)
G1 = 4*pi/sqrt(3)/a0*np.array([0, 1])
G2 = Rot(pi/3).dot(G1)
g1 = theta*np.array([G1[1], -G1[0]])
g2 = theta*np.array([G2[1], -G2[0]])
kD = g1[0]/sqrt(3)
Kb = np.array([-kD*sqrt(3)/2, -kD/2])

#define Lattice
L = []
invL = np.zeros((2*N+1, 2*N+1), int)
def Lattice(n):
    count = 0
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])
            invL[i+n, j+n] = count
            count = count + 1
Lattice(N)
siteN = (2*N+1)**2
L = np.array(L)

def SolvHamiltonian(kx, ky):
    H = np.zeros((siteN, siteN), dtype=complex)
    for i in np.arange(siteN):
        ix = L[i, 0]
        iy = L[i, 1]
        ax = kx - Kb[0] + ix*g1[0] + iy*g2[0]
        ay = ky - Kb[1] + ix*g1[1] + iy*g2[1]
        H[i, i] += kin * (ax**2 + ay**2)
        if (ix != N):
            j = invL[ix+1 +N, iy +N]
            H[j, i] += V*exp(I*psi)
        if (iy != N):
            j = invL[ix +N, iy+1 +N]
            H[j, i] += V*exp(-I*psi)
        if ((ix != -N) and (iy != N)):
            j = invL[ix-1 +N, iy+1 +N] 
            H[j, i] += V*exp(I*psi)
        if (ix != -N):
            j = invL[ix-1 +N, iy +N]
            H[j, i] += V*exp(-I*psi)
        if (iy != -N):
            j = invL[ix +N, iy-1 +N]
            H[j, i] += V*exp(I*psi)
        if ((ix != N) and (iy != -N)):
            j = invL[ix+1 +N, iy-1 +N] 
            H[j, i] += V*exp(-I*psi)
    eigenE = np.linalg.eigvalsh(H)
    e = np.sort(eigenE)
    return e

Num = 50
KX = list(np.linspace(0,0,Num)) + list(np.linspace(0,kD/2*sqrt(3),Num)) + \
     list(np.linspace(kD/2*sqrt(3), 0, Num*2))
KY = list(np.linspace(kD,0,Num)) + list(np.linspace(0,kD/2,Num)) + \
     list(np.linspace(kD/2,-kD, 2*Num))
Eigen = np.zeros((len(KX),siteN))
for k in range(len(KX)):
    Eigen[k,:] = SolvHamiltonian(KX[k], KY[k])
plt.plot(np.linspace(0,1,len(KX)),Eigen,'b-')
plt.ylim(-0.060,0.010)
#plt.xlim(0,1)
#plt.xticks([])
#plt.vlines(1/5,0,0.025,color = "k")
#plt.vlines(2/5,0,0.025,color = "k")
#plt.vlines(3/5,0,0.025,color = "k")
#plt.xticks([0,1/5,1/5*2,1/5*3,1],['k+"',r'$\gamma$','k-','k+','k+"'])

plt.show()