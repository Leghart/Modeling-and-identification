import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from numpy import random
from numpy.linalg import inv, norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

########## zad 1 + 2 #################################
a = 0.1
b = 0.1

N = 100


teta = np.array([b, a])

# wejscie systemu 
U = random.normal(0, 1, N)

# zaklocenie biale
#e = np.array(random.normal(0, 1, N)) *0


def Imp(teta):
    U = np.zeros(N)
    U = np.insert(U,0,1)
    U = U[:-1]
    Yoff = 0
    Yn = []
    for i in range(N):
        fi = np.array([U[i],Yoff])
        Yoff = fi.T @ teta
        Yn.append(Yoff)
    return Yn

teta_list= [[0.5,0.5],[2,0.5],[0.5,1.2]]

fig, axs = plt.subplots(1,3, figsize=(15,5))
for i in range(3):
    axs[i].plot(Imp(teta_list[i]))
    axs[i].grid()
    axs[i].set_title('a = {}, b = {}'.format(teta_list[i][1],teta_list[i][0]))


############# zad 3 4 #############################
Un = np.random.uniform(0, 1, N)
c = 1
en = np.random.uniform(-c, c, N)



def System(Un, en, teta, a):
    Zn = []
    Res = 0
    Phi = []
    e_s = 0
    
    for i in range(len(Un)):
        fi = np.array([Un[i], Res])
        Phi.append(fi)
        Res = fi.T @ teta + en[i]
        Zn.append(en[i] - a*e_s)
        e_s = en[i]
        
    Zn = np.array(Zn)
    Phi = np.matrix(Phi)
    
    return (Phi @ teta + Zn.T).T , Phi

Yn, Phi = System(Un, en, teta, a)


def MNK(Phi, Y):
    return inv(Phi.transpose() @ Phi) @ Phi.transpose() @ Y


               
teta_est = np.array(MNK(Phi,Yn))

plt.figure(figsize=(10,5))
plt.plot(teta, '.', label='$\Theta$')
plt.plot(teta_est, 'x', label='$\hat{\Theta}$')
plt.grid()
plt.legend()

############# zad 5 ################
def Err_MNK(L, N, teta, a):
    suma = 0
    for l in range(L):
        Un = np.random.uniform(0,1,N)
        en = np.random.uniform(-1, 1, N)
        Yn, Phi = System(Un, en, teta, a)
        teta_est = MNK(Phi, Yn)
        suma += np.linalg.norm(teta_est - teta)**2
    return suma/L

L = 100
Err_MNK_list = []

for n in range(10,500):
    Err_MNK_list.append(Err_MNK(L,n,teta,a))

plt.figure(figsize=(10,5))
plt.plot(list(range(10,500)), Err_MNK_list)
plt.grid()
plt.xlabel('N')
plt.ylabel('Err')

############# zad 6 #################
Vn_est = []
pom = 0 
for i in range(N):
    Vn_est.append(teta_est[0][0] * Un[i] + teta_est[1][0] * pom)
    pom = Vn_est[i]
    
########## zad 7 ##############
Psi = np.array([Un, Vn_est]).T

######### zad 8 #################
def IV(Psi,Phi,Y):
    return inv(Psi.T @ Phi) @ Psi.T @ Y

teta_IV = IV(Psi,Phi,Yn)


plt.figure(figsize=(10,5))
plt.plot(teta, '.', label='$\Theta$')
plt.plot(teta_IV, 'x', label='$\hat{\Theta}$')
plt.grid()
plt.legend()

####### zad 9 #################

def Err_IV(L, N, teta, a):
    suma = 0
    for l in range(L):
        
        Un = np.random.uniform(0,1,N)
        en = np.random.uniform(-1, 1, N)
        Yn, Phi = System(Un, en, teta, a)
        teta_est = MNK(Phi, Yn)
        
        Vn_est = []
        pom = 0 
        for i in range(N):
            tmp = teta_est[0][0] * Un[i] + teta_est[1][0] * pom
            Vn_est.append(tmp)
            pom = Vn_est[i]

        Vn_est = np.squeeze(np.array(Vn_est))
        

        Psi = np.array([Un, Vn_est]).T
        teta_IV = inv(Psi.T @ Phi) @ Psi.T @ Yn
        suma += np.linalg.norm(teta_IV - teta)**2
        
    return suma/L


L = 50
Err_IV_list = []
Err_MNK_list = []

for n in range(10,500):
    Err_IV_list.append(Err_IV(L,n,teta,a))
    Err_MNK_list.append(Err_MNK(L,n,teta,a))
    

plt.figure(figsize=(10,5))
plt.plot(list(range(10,500)), Err_IV_list, label='IV')
plt.plot(list(range(10,500)), Err_MNK_list, label='MNK')
plt.grid()
plt.xlabel('N')
plt.ylabel('Err')
plt.legend()
