import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from numpy import random
from numpy.linalg import inv, norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

########## zad 1 + 2 #################################
N = 1000
sig = 2
U = random.normal(0, sig**2, N)
s = 10
b = [0,-2,-10,4,-8,1,-8,3,-10,1]
Zn = np.array(random.normal(0, 1, N))


U = np.concatenate((np.zeros(s-1),U), axis=0)

Phi = np.array([U[i-9:i+1] for i in range(9,N+9)])

Y = Phi @ np.array(b) + Zn

def MNK(X,Y):
    return inv((X.transpose() @ X)) @ X.transpose() @ Y

est = MNK(Phi,Y)
'''
plt.figure(figsize=(10,5))
plt.plot(b, '.', label='$b^*$')
plt.plot(est, 'o', fillstyle='none', label = '$\hat{b}_N$')
plt.legend()
'''
################# zad 3 ###############
def Err(a_est, b):
    L = len(a_est)
    return sum([norm(a_est[l] - b)**2 for l in range(L)]) / L

'''
L = 100
b_est = []
out = []
sig = 1
for nn in range(100,1000,10):
    b_est.clear()
    U = np.concatenate((np.zeros(s-1),random.normal(0, sig**2, nn)), axis=0)
    Phi = np.array([U[i-9:i+1] for i in range(9,nn+9)])
    for i in range(L):
        Z = random.normal(0, 1, nn)
        Y = Phi @ np.array(b) + Z
        b_est.append(MNK(Phi, Y))
    out.append(Err(b_est, b))
plt.figure(figsize=(10,5))
plt.plot(list(range(100,1000,10)),out)
plt.grid(which='both')
plt.xlabel('N')
plt.ylabel('Err')
plt.legend()

'''
########## zad 4 #####################
def gen_Z(e, alfa):
    return [e[i] + alfa * e[i-1] for i in range(len(e))]

'''
sig = 2
L = 100
b_est = []
b_est1 = []
out= []
out1 = []
for nn in range(100,1000,10):
    b_est.clear()
    b_est1.clear()
    U = np.concatenate((np.zeros(s-1),random.normal(0, sig**2, nn)), axis=0)
    Phi = np.array([U[i-9:i+1] for i in range(9,nn+9)])
    for i in range(L):
        # skorel
        alfa = 0.5
        Z = gen_Z(random.normal(0, sig**2, nn), alfa)
        Y = Phi @ np.array(b) + Z
        b_est.append(MNK(Phi, Y))
        
        # bialy
        Z = random.normal(0, sig**2, nn)
        Y = Phi @ np.array(b) + Z
        b_est1.append(MNK(Phi, Y))
    
    out.append(Err(b_est, b))
    out1.append(Err(b_est1, b))

plt.figure(figsize=(10,5))  
plt.plot(list(range(100,1000,10)),out, label='szum skorelowany')
plt.plot(list(range(100,1000,10)),out1, label='szum bialy')
plt.grid(which='both')
plt.xlabel('N')
plt.ylabel('Err')
plt.legend()
'''

########## zad 5 #####################################
sig = 1
alfa = 0.5
c0 = (1 + alfa**2)*sig**2
c1 = alfa*sig**2

from scipy.linalg import toeplitz

R_col = np.concatenate(([c0,c1],np.zeros(N-2)), axis=0)
R = toeplitz(R_col)
print('Det(R) = {}'.format(np.linalg.det(R)))


########## zad 6 ####################
def b_est_GLS(Phi, Y, R):
    return inv(Phi.transpose() @ inv(R) @ Phi) @ Phi.transpose() @ inv(R) @ Y

#print('Estymator GLS = {}'.format(b_est_GLS(Phi,Y,R)))




sig = 1
L = 20
alfa = 0.5
b_est = []
b_est1 = []
b_bialy = []
out= []
out1 = []
out2=[]
for nn in range(100,1000,10):
    b_est.clear()
    b_est1.clear()
    b_bialy.clear()
    U = np.concatenate((np.zeros(s-1),random.normal(0, sig**2, nn)), axis=0)
    Phi = np.array([U[i-9:i+1] for i in range(9,nn+9)])
    for i in range(L):
        
        # skorel
        Z = gen_Z(random.normal(0, sig**2, nn), alfa)
        Y = Phi @ np.array(b) + Z
        b_est.append(MNK(Phi, Y))
        
        # bialy
        Z = random.normal(0, sig**2, nn)
        Y = Phi @ np.array(b) + Z
        b_est1.append(MNK(Phi, Y))
        
        #gls
        R_col = np.concatenate(([c0,c1],np.zeros(nn-2)), axis=0)
        R = toeplitz(R_col)
        b_bialy.append(b_est_GLS(Phi,Y,R))
        
        
    out.append(Err(b_est, b))
    out1.append(Err(b_est1, b))
    out2.append(Err(b_bialy, b))

plt.figure(figsize=(10,5))  
plt.plot(list(range(100,1000,10)),out, label='szum skorelowany')
plt.plot(list(range(100,1000,10)),out1, label='szum bialy')
plt.plot(list(range(100,1000,10)),out2, label='wybielony szum skorelowany przy uzyciu GLS')
plt.grid(which='both')
plt.xlabel('N')
plt.ylabel('Err')
plt.legend()


















######## zad 7 #################################
'''
# dla szumu skorelowanego
L = 20
b_est = []
b_est_gls = []
b_est_bialy = []
out= []
out1 = []
out2=[]
sig = 1
for nn in range(100,1000,10):
    b_est.clear()
    b_est_gls.clear()
    U = np.concatenate((np.zeros(s-1),random.normal(0, sig**2, nn)), axis=0)
    Phi = np.array([U[i-9:i+1] for i in range(9,nn+9)])
    for i in range(L):
        # skorel zwykly est
        Z = gen_Z(random.normal(0, sig**2, nn), alfa)
        Y = Phi @ np.array(b) + Z
        b_est.append(MNK(Phi, Y))
        
        # skorel z gls
        R_col = np.concatenate(([c0,c1],np.zeros(nn-2)), axis=0)
        R = toeplitz(R_col)
        b_est_gls.append(b_est_GLS(Phi,Y,R))
        
    
    out.append(Err(b_est, b))
    out1.append(Err(b_est_gls, b))


plt.figure(figsize=(10,5))  
plt.plot(list(range(100,1000,10)),out, label='MNK dla skorelowanego')
plt.plot(list(range(100,1000,10)),out1, label='wybielenie przy uzyciu GLS')
plt.grid(which='both')
plt.xlabel('N')
plt.ylabel('Err')

plt.legend()

'''

'''
# dla szumu bialego
L = 20
b_est = []
b_est_gls = []
out= []
out1 = []
for nn in range(100,1000,10):
    b_est.clear()
    b_est_gls.clear()
    U = np.concatenate((np.zeros(s-1),random.normal(0, sig**2, nn)), axis=0)
    Phi = np.array([U[i-9:i+1] for i in range(9,nn+9)])
    for i in range(L):
        # skorel zwykly est
        Z = np.array(random.normal(0, 1, nn))
        #Z = gen_Z(random.normal(0, sig**2, nn), alfa)
        Y = Phi @ np.array(b) + Z
        b_est.append(MNK(Phi, Y))
        
        # skorel z gls
        R_col = np.concatenate(([c0,c1],np.zeros(nn-2)), axis=0)
        R = toeplitz(R_col)
        b_est_gls.append(b_est_GLS(Phi,Y,R))
    
    out.append(Err(b_est, b))
    out1.append(Err(b_est_gls, b))

plt.figure(figsize=(10,5))  
plt.plot(list(range(100,1000,10)),out, label='estymator MNK')
plt.plot(list(range(100,1000,10)),out1, label='estymator GLS')
plt.grid(which='both')
plt.xlabel('N')
plt.ylabel('Err')

plt.legend()

'''
