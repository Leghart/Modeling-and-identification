import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy.linalg import inv, norm

########## zad 1 + 2 #################################
N = 1000
sig = 1
#D = int(N**(1/3))
D = int(N**(1/3))
mu = np.ones(D)
a = random.randint(0,10,D)

#a = [0,-2,-10,4,-8,1,-8,3,-10,1,-3,-7,5,3,-3,1,-8,2,2,-2]


E = np.eye(D) * sig**2
Z = random.normal(0, sig**2, N)
Xn = random.multivariate_normal(mu, E, N)
Y = [Xn[n] @ a + Z[n] for n in range(N)]

'''
plt.figure(figsize=(10,5))
for i in range(D):   
    plt.plot(Xn[:,i],Y,'.',label='wejscie {}'.format(i+1))

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

'''
############ zad 3 ###################################
def MNK(X,Y):
    return inv((X.transpose() @ X)) @ X.transpose() @ Y


'''
#fig,axs = plt.subplots(1,2,figsize=(10,5))
plt.figure(figsize=(10,5))
i = 0
#DD = [int(N**(1/3)), int(N**(1/2))]
DD = [N]
for dd in DD:
    D = dd
    a = random.randint(0,N,D)
    E = np.eye(D) * 1**2
    Z = random.normal(0, 1**2, N)
    mu = np.ones(D)
    Xn = random.multivariate_normal(mu, E, N)
    Y = [Xn[n] @ a + Z[n] for n in range(N)]
    
    est = MNK(Xn,Y)
    plt.plot(a, '.', label='$a^*$')
    plt.plot(est, 'o', fillstyle='none', label = '$\hat{a}_N$')
    plt.grid()
    plt.legend()
    i+=1
'''
'''
fig,axs = plt.subplots(1,3,figsize=(10,5))
sig = [1]
for i,s in enumerate(sig):
    D = int(N**(1/3))
    E = np.eye(D) * s**2
    Z = random.normal(0, s**2, N)
    mu = np.ones(D)
    Xn = random.multivariate_normal(mu, E, N)
    Y = [Xn[n] @ a + Z[n] for n in range(N)]
    
    est = MNK(Xn,Y)

    axs[i].plot(a, '.', label='$a^*$')
    axs[i].plot(est, 'o', fillstyle='none', label = '$\hat{a}_N$')
    axs[i].legend()
    axs[i].set_title('$\sigma_N=${}'.format(s))
'''

########## zad 4 #####################################
def cov(X):
    return sig**2 * inv((X.transpose() @ X))

'''
D = int(N**(1/3))
DD = [N]
plt.figure(figsize=(10,5))
i= 0
for D in DD:
    mu = np.ones(D)
    a = random.randint(0,10,D)
    E = np.eye(D) * sig**2
    Xn = random.multivariate_normal(mu, E, N)
    
    plt.pcolor(cov(Xn))
    i+=1
'''    
########## zad 5 #####################################
def Err(a_est):
    L = len(a_est)
    return sum([norm(a_est[l] - a)**2 for l in range(L)]) / L


'''
plt.figure(figsize=(10,5))
L = 20
a_est = []
out = []
w= 0
SIG = [1]
#DD = [int(N**(1/3)), int(N**(1/2))]
DD = [N]
#stri = ['$N^{1/3}$', '$N^{1/2}$']
stri = ['$N$']
for dd in DD:
    out.clear()
    for nn in range(100,1000,10):
        a_est.clear()
        mu = np.ones(dd)
        a = random.randint(0,10,dd)
        E = np.eye(dd) * 1**2
        Xn = random.multivariate_normal(mu, E, nn)
        for i in range(L):
            Z = random.normal(0, 1**2, nn)
            Y = [Xn[n] @ a + Z[n] for n in range(nn)]
            a_est.append(MNK(Xn, Y))
        out.append(Err(a_est))
    plt.plot(list(range(100,1000,10)),out,label='D = {}'.format(stri[w]))
    w+=1
plt.grid()
plt.xlabel('N')
plt.ylabel('Err')
plt.legend()
'''
