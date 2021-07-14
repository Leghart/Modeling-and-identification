import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from numpy import random
from numpy.linalg import inv, norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

########## zad 1 + 2 #################################
N = 100
sig = 1
D = 20
mu = np.ones(D)
a = random.randint(0,10,D)
#a = [0,-2,-10,4,-8,1,-8,3,-10,1,-3,-7,5,3,-3,1,-8,2,2,-2]
#a = [-1,5,]


E = np.eye(D) * sig**2

e = random.normal(0, sig**2, N)

def gen_Z(e, b):
    Zn = []
    for i in range(len(e)):
        Zn.append(e[i] + b * e[i-1])
    return Zn

Z = gen_Z(e, 2)

Xn = random.multivariate_normal(mu, E, N)
Y = [Xn[n] @ a + Z[n] for n in range(N)]


############ zad 3 ###################################
def MNK(X,Y):
    return inv((X.transpose() @ X)) @ X.transpose() @ Y

'''
fig,axs = plt.subplots(1,3,figsize=(10,5))
sig = [3,5,10]
for i,s in enumerate(sig):
    E = np.eye(D) * s**2
    Z = random.normal(0, s**2, N)
    Xn = random.multivariate_normal(mu, E, N)
    Y = [Xn[n] @ a + Z[n] for n in range(N)]
    
    est = MNK(Xn,Y)

    axs[i].plot(a, '.', label='$a^*$')
    axs[i].plot(est, 'o', fillstyle='none', label = '$\hat{a}_N$')
    axs[i].legend()
    axs[i].set_title('$\sigma_N=${}'.format(s))

'''
########## zad 5 #####################################

def cov(X, R):
    return  inv((X.transpose() @ X)) @ X.transpose() @ R @ X @ inv((X.transpose() @ X))


def stare_cov(X):
    return sig**2 * inv((X.transpose() @ X))


ZZ = [gen_Z(random.normal(0, sig**2, N),1) for _ in range(N)]
WC = np.cov(ZZ)


fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
im1 = ax1.imshow(stare_cov(Xn), interpolation='None')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='horizontal')
ax1.set_title('A: $\sigma^2_N=${}'.format(sig**2))

ax2 = fig.add_subplot(122)
im2 = ax2.imshow(cov(Xn,WC), interpolation='None')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='horizontal');
ax2.set_title('B: $\sigma^2_N=${}'.format(sig**2))

############################# Badanie zmiany B ########################
'''
sig = 1
b = 1
ZZ = [gen_Z(random.normal(0, sig**2, N),b) for _ in range(N)]
R = np.cov(ZZ)

fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(121)
im1 = ax1.imshow(cov(Xn,R))

divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='horizontal')
ax1.set_title('A: b = {}'.format(b))


b = 3
ZZ = [gen_Z(random.normal(0, sig**2, N),b) for _ in range(N)]
R = np.cov(ZZ)
ax2 = fig.add_subplot(122)
im2 = ax2.imshow(cov(Xn,R), interpolation='None')


divider = make_axes_locatable(ax2)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='horizontal');
ax2.set_title('B: b = {}'.format(b))

b = 5
ZZ = [gen_Z(random.normal(0, sig**2, N),b) for _ in range(N)]
R = np.cov(ZZ)

fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(121)
im1 = ax1.imshow(cov(Xn,R))

divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='horizontal')
ax1.set_title('A: b = {}'.format(b))


b = 10
ZZ = [gen_Z(random.normal(0, sig**2, N),b) for _ in range(N)]
R = np.cov(ZZ)
ax2 = fig.add_subplot(122)
im2 = ax2.imshow(cov(Xn,R), interpolation='None')


divider = make_axes_locatable(ax2)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='horizontal');
ax2.set_title('B: b = {}'.format(b))
'''
'''
########## zad 5 #####################################
def Err(a_est):
    L = len(a_est)
    return sum([norm(a_est[l] - a)**2 for l in range(L)]) / L


#plt.figure(figsize=(10,5))
fig,axs = plt.subplots(1,2,figsize=(10,5))

L = 20
a_est = []
out = []
SIG = [1,2,3,4]

for ss in SIG:
    out.clear()
    for nn in range(100,1000,10):
        a_est.clear()
        Xn = random.multivariate_normal(mu, E, nn)
        for i in range(L):
            Z = random.normal(0, ss**2, nn)
            Y = [Xn[n] @ a + Z[n] for n in range(nn)]
            a_est.append(MNK(Xn, Y))
        out.append(Err(a_est))
    axs[0].plot(list(range(100,1000,10)),out,label='$\sigma_N^2$={}'.format(ss**2))
    axs[0].grid(which='both')
    axs[0].set_xlabel('N')
    axs[0].set_ylabel('Err')
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title('Zakłócenie i.i.d.')

for ss in SIG:
    out.clear()
    for nn in range(100,1000,10):
        a_est.clear()
        Xn = random.multivariate_normal(mu, E, nn)
        for i in range(L):
            e = random.normal(0, ss**2, N)
            b = 2
            Z = gen_Z(e, b)
            Y = [Xn[n] @ a + Z[n] for n in range(nn)]
            a_est.append(MNK(Xn, Y))
        out.append(Err(a_est))

    axs[1].plot(list(range(100,1000,10)),out,label='$\sigma_N^2$={}'.format(ss**2))
    axs[1].grid(which='both')
    axs[1].set_xlabel('N')
    axs[1].set_ylabel('Err')
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title('Zakłócenie skorelowane - b = {}'.format(b))

'''
##################################
'''
def dod_Zn(e, ff, K):
    lamb = np.array([ff**k for k in range(K)])
    Zn = []
    pom = (np.zeros(len(lamb)))

    for i in range(len(e)):
        pom = np.insert(pom, 0, e[i])
        pom = pom[:-1]
        Zn.append(sum(lamb*pom))
    return Zn

N = 80

ZZ = [dod_Zn(random.normal(0, sig**2, N), 0.5, 15) for _ in range(1000)]
R = np.cov(np.transpose(ZZ))
print('Skorelowane: ', np.linalg.det(R))


fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(121)
lam = 0.1
k = 4
ZZ = [dod_Zn(random.normal(0, sig**2, N), lam, k) for _ in range(1000)]
im1 = ax1.imshow(np.cov(np.transpose(ZZ)), interpolation='None')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='horizontal')
ax1.set_title('$\lambda$ = {}, K = {}'.format(lam, k))

ax2 = fig.add_subplot(122)
lam = 0.25
k = 8
ZZ = [dod_Zn(random.normal(0, sig**2, N), lam, k) for _ in range(1000)]
im2 = ax2.imshow(np.cov(np.transpose(ZZ)), interpolation='None')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='horizontal')
ax2.set_title('$\lambda$ = {}, K = {}'.format(lam, k))



fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
lam = 0.5
k = 10
ZZ = [dod_Zn(random.normal(0, sig**2, N), lam, k) for _ in range(1000)]
im1 = ax1.imshow(np.cov(np.transpose(ZZ)), interpolation='None')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='horizontal')
ax1.set_title('$\lambda$ = {}, K = {}'.format(lam, k))

ax2 = fig.add_subplot(122)
lam = 0.98
k = 15
ZZ = [dod_Zn(random.normal(0, sig**2, N), lam, k) for _ in range(1000)]
im2 = ax2.imshow(np.cov(np.transpose(ZZ)), interpolation='None')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='horizontal')
ax2.set_title('$\lambda$ = {}, K = {}'.format(lam, k))
'''
