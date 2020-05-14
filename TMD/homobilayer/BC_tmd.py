'''
Author: zihaophys
Description: This is Berry curvature for the first band in tight-binding 
model of twisted TMD, which is fitted from continuum model
'''

import numpy as np 
import matplotlib.pyplot as plt
from numpy import *
t0   = 0.29
t1   = 0.06
I    = complex(0,1)
t120 = np.exp(I*2*np.pi/3)
t240 = np.exp(-I*2*np.pi/3)

def BC(kx, ky):
    H = np.zeros((2, 2), dtype=complex)
    h = t1*t120*( np.exp(I*ky) + np.exp(-I*(sqrt(3)/2*kx+1/2*ky)) + np.exp(I*(sqrt(3)/2*kx-1/2*ky)) )
    hp = t1*t240*( np.exp(I*ky) + np.exp(-I*(sqrt(3)/2*kx+1/2*ky)) + np.exp(I*(sqrt(3)/2*kx-1/2*ky)) )
    tu = t0*np.exp(-I*kx/sqrt(3)) * (1+np.exp(I*(sqrt(3)/2*kx+1/2*ky))+np.exp(I*(sqrt(3)/2*kx-1/2*ky)))
    
    H[0, 0] = h + h.conjugate()
    H[1, 1] = hp + hp.conjugate()
    H[0, 1] = tu
    H[1, 0] = tu.conjugate()

    pxH = np.zeros((2, 2), dtype=complex)

    pxh = t1*t120*I*sqrt(3)/2*( np.exp(I*(sqrt(3)/2*kx-1/2*ky)) - np.exp(-I*(sqrt(3)/2*kx+1/2*ky)))
    pxH[0, 0] = pxh + pxh.conjugate()
    pxhp = t1*t240*I*sqrt(3)/2*( np.exp(I*(sqrt(3)/2*kx-1/2*ky)) - np.exp(-I*(sqrt(3)/2*kx+1/2*ky)))
    pxH[1, 1] = pxhp + pxhp.conjugate()
    pxH[0, 1] = t0 * ( -I/sqrt(3) *  np.exp(-I*kx/sqrt(3)) * (1+np.exp(I*(sqrt(3)/2*kx+1/2*ky))+np.exp(I*(sqrt(3)/2*kx-1/2*ky))) \
        + np.exp(-I*kx/sqrt(3))*I*sqrt(3)/2 *(np.exp(I*(sqrt(3)/2*kx+1/2*ky))+np.exp(I*(sqrt(3)/2*kx-1/2*ky))) )
    pxH[1, 0] = pxH[0, 1].conjugate()

    pyH = np.zeros((2, 2), dtype=complex)
    pyh = t1*t240*I/2*( np.exp(-I*(sqrt(3)/2*kx-1/2*ky)) + np.exp(I*(sqrt(3)/2*kx+1/2*ky)) - 2*np.exp(-I*ky) )
    pyH[0, 0] = pyh + pyh.conjugate()
    pyhp = t1*t120*I/2*( np.exp(-I*(sqrt(3)/2*kx-1/2*ky)) + np.exp(I*(sqrt(3)/2*kx+1/2*ky)) - 2*np.exp(-I*ky) )
    pyH[1, 1] = pyhp + pyhp.conjugate()
    pyH[0, 1] = t0*np.exp(-I*kx/sqrt(3))*I/2* ( np.exp(I*(sqrt(3)/2*kx+1/2*ky))-np.exp(I*(sqrt(3)/2*kx-1/2*ky)) )
    pyH[1, 0] = pyH[0, 1].conjugate()

    eigenE, eigenV = np.linalg.eig(H)
    eigenE_sorted = np.sort(eigenE)
    eigenV_sorted = eigenV[:, eigenE.argsort()]
    up = eigenV_sorted[:,1].T
    down = eigenV_sorted[:,0].T
    Numerator = ( up.conjugate().T.dot(pxH).dot(down) ) * ( down.conjugate().T.dot(pyH).dot(up) ) - \
            ( up.conjugate().T.dot(pyH).dot(down) ) * ( down.conjugate().T.dot(pxH).dot(up) )
    Denominator = (eigenE_sorted[0]-eigenE_sorted[1])**2
    return -(Numerator/Denominator).imag
    #return eigenE_sorted[0]

kD = 4*np.pi/3
alphalist = np.arange(-sqrt(3)/2, sqrt(3)/2+0.1, 0.01)
betalist  = np.arange(-1, 1.2, 0.01)

kkx = len(alphalist)
kky = len(betalist)
E = np.zeros((kky, kkx))

for kx in range(kkx):
    for ky in range(kky):
 
        x= (alphalist[kx])
        y = (betalist[ky]) 
         
        E[ky][kx] = BC(x*kD, y*kD)
 

plt.contour(alphalist, betalist, E, levels=[0],colors='k')

plt.contourf(alphalist, betalist, E, 50, cmap='viridis')
plt.ylim(-1,1)
plt.xlim(-sqrt(3)/2, sqrt(3)/2)
plt.colorbar()
plt.axes().set_aspect('equal')
plt.show()
