'''
Author:zihaophys
Description: This is for TMDC continuum model with valley degeneracy
4 by 4 model
'''


import numpy as np 
import matplotlib.pyplot as plt
from numpy import *
t0   = 0.29
t1   = 0.06
I    = complex(0,1)
t120 = np.exp(I*2*np.pi/3)
t240 = np.exp(-I*2*np.pi/3)

def Hamiltonian(kx, ky):
    H = np.zeros((4, 4), dtype=complex)

    h = t1*t120*( np.exp(I*ky) + np.exp(-I*(sqrt(3)/2*kx+1/2*ky)) + np.exp(I*(sqrt(3)/2*kx-1/2*ky)) )
    hp = t1*t240*( np.exp(I*ky) + np.exp(-I*(sqrt(3)/2*kx+1/2*ky)) + np.exp(I*(sqrt(3)/2*kx-1/2*ky)) )

    H[0, 0] = h + h.conjugate()
    H[3, 3] = H[0, 0]
    H[1, 1] = hp + hp.conjugate()
    H[2, 2] = H[1, 1]

    tu = t0*np.exp(-I*kx/sqrt(3)) * (1+np.exp(I*(sqrt(3)/2*kx+1/2*ky))+np.exp(I*(sqrt(3)/2*kx-1/2*ky)))
    H[0, 2] = tu
    H[1, 3] = H[0, 2]
    H[2, 0] = tu.conjugate()
    H[3, 1] = H[2, 0]

    eigenvalue,featurevector=np.linalg.eig(H)
    eig_vals_sorted = np.sort(eigenvalue)
    e=eig_vals_sorted
    return e

kD = 4*np.pi/3
KtoG = np.arange(1, 0, -1/100)
GtoKp = np.arange(0, 1, 1/100)
KptoK = np.arange(1/2, -1/2, -1/100)
KptoKp = np.arange(-sqrt(3)/2, sqrt(3)/2, 1/100)
AllK = len(KtoG) + len(GtoKp) + len(KptoK) + len(KptoKp)
E  = np.zeros((AllK,4), float)

for i in np.arange(0, len(KtoG)):
    k = KtoG[i]
    E[i] = np.real(Hamiltonian(-k*sqrt(3)/2*kD, -k/2*kD))
for i in np.arange(len(KtoG), len(KtoG)+len(GtoKp)):
    k = GtoKp[i-len(KtoG)]
    E[i] = np.real(Hamiltonian(-k*sqrt(3)/2*kD, k/2*kD))
for i in np.arange(len(KtoG)+len(GtoKp), len(KtoG)+len(GtoKp)+len(KptoK)):
    k = KptoK[i-len(KtoG)-len(GtoKp)]
    E[i] = np.real(Hamiltonian(-sqrt(3)/2*kD, k*kD))
for i in np.arange(len(KtoG)+len(GtoKp)+len(KptoK), AllK):
    k = KptoKp[i-len(KtoG)-len(GtoKp)-len(KptoK)]
    E[i] = np.real(Hamiltonian(k*kD, -1/2*kD))
for j in range(0,4):
    plt.plot(np.arange(AllK), E[:,j], linestyle="-", linewidth=2)


plt.show()