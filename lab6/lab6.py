import matplotlib.pyplot as plt
import random as rand
from numpy import random
import numpy as np
import math
import time
from scipy import integrate


def gen_roz(fun,N,a,b):
    x=np.arange(a,b,0.001) # wektor próbek nosnika
    y=np.zeros(len(x))

    # obliczenie wartosci funkcji w punktach z nosnika
    for i in range(len(x)):
        y[i]=fun(x[i]) 
    
    # ograniczenie górne jako maksymalna wartosc funkcji
    d=np.ceil(max(y)) 
    
    X=np.empty(0)

    # generacja N zmiennych losowych 
    while(len(X)<N):  
        u1=np.random.uniform(a,b)
        u2=np.random.uniform(0,d)
        if u2<=fun(u1):
            X=np.append(X,u1)
    return X

def I(x):
    if abs(x)<=1:
        return 1
    elif abs(x)>1:
        return 0

def f(mu,sig,x):
    return 1/(sig*math.sqrt(2*math.pi))*np.exp(-(x-mu)**2/(2*sig**2))

# jadro prostokatne
def K1(x):
    return 0.5*I(x)

# jadro gausowskie
def K2(x):
    return math.exp(-x**2/2)/(math.sqrt(2*math.pi))

# jadro epanechnikova
def K3(x):
    return 3/4*(1-x**2)*I(x)

# jadro tricube
def K4(x):
    return 70/81*(1-abs(x)**3)**3*I(x)

# estymator jadrowy
def est_jadr(X,x,hN,K):
     return sum([K((i-x)/hN) for i in X])/(N*hN)




################# PANEL STEROWANIA ###########################

BINS=55
WIDTH=0.7

############################ zad 1 ###############################
N=1000

wsp = 2
'''
def m(x):
    return math.atan(x*a)
'''

def m(x):
    if 0 <= abs(x) and abs(x) < 1:
        return wsp*x**2
    elif 1 <= abs(x) and abs(x) < 2:
        return wsp
    elif 2 <= abs(x) and abs(x) < 10000:
        return 0


N = 5000
X = np.random.uniform(-5,5,N)
#X = np.random.normal(2,1,N)
sig = 1
Z=np.random.normal(0,sig,N)

############################ zad 2 ###############################
Y=[m(X[i])+Z[i] for i in range(len(X))]
x=np.linspace(-5,5,N)
SYS=[m(i) for i in x]

#plt.plot(X,Y,'.')
#plt.plot(x,SYS,'.')

############################### zad 3 #############################
def m_est(x,X,Y,hN,K):
    return sum([Y[i]*K((X[i]-x)/hN) for i in range(len(X))])/sum([K((X[i]-x)/hN) for i in range(len(X))])
    

m_oryg=[m(i) for i in x]
m_estym=[m_est(i,X,Y,0.3,K1) for i in x]

plt.plot(x,m_oryg,label='oryginalny system')
plt.plot(x,m_estym,label='estymowany system')
plt.legend()
plt.grid()

'''
hN=[0.1,0.9,1.5]
fig,axs=plt.subplots(1,3,figsize=(15,5))
for h in range(len(hN)):
    m_oryg=[m(i) for i in x]
    m_estym=[m_est(i,X,Y,hN[h],K1) for i in x]
    
    axs[h].plot(x,m_oryg,label='oryginalny system')
    axs[h].plot(x,m_estym,label='estymowany system')
    axs[h].legend()
    axs[h].set_title('$h_N = $'+str(hN[h]))
    axs[h].grid()
'''
################################# zad 4 ############################
"""
hN=1
a=1
x=np.linspace(-2,2,N)
Y=[m(X[i])+Z[i] for i in range(len(X))]

K=[K1,K2,K3,K4]
m_oryg=[m(i) for i in x]
plt.figure(figsize=(10,5))
plt.plot(x,m_oryg,label='oryginalny system')
jadra=['Jadro prostokatne','Jadro Gaussowskie','Jadro Epanechnikova','Jadro Tricube']
for k in range(len(K)):
    m_estym = [m_est(i, X, Y, hN, K[k]) for i in x]
    plt.plot(x,m_estym, label='{}'.format(jadra[k]))
plt.legend()
plt.title('a = ' + str(a)) 
plt.grid()



hN=1
a=10
x=np.linspace(-2,2,N)
Y=[m(X[i])+Z[i] for i in range(len(X))]

K=[K1,K2,K3,K4]
m_oryg=[m(i) for i in x]
plt.figure(figsize=(10,5))
plt.plot(x,m_oryg,label='oryginalny system')
jadra=['Jadro prostokatne','Jadro Gaussowskie','Jadro Epanechnikova','Jadro Tricube']
for k in range(len(K)):
    m_estym = [m_est(i, X, Y, hN, K[k]) for i in x]
    plt.plot(x,m_estym, label='{}'.format(jadra[k]))
plt.legend()
plt.title('a = ' + str(a)) 
plt.grid()

"""
############################### zad 5 ###############################

def valid(X,Y,h,K):
    Q=100
    suma=sum([(m_est(q/Q,X,Y,h,K)-m(q/Q))**2 for q in range(-Q,Q,1)])
    return suma/(2*Q)
    

'''
hn=np.arange(0.05,1,0.01)
wek=[valid(X,Y,h,K1) for h in hn]
idx=wek.index(min(wek))
hmin=hn[idx]

'''

############################### zad6 ###############################
'''
# p3
m_oryg=[m(i) for i in x]
m_estym=[m_est(i,X,Y,hmin,K1) for i in x]

plt.plot(x,m_oryg,label='oryginalny system')
plt.plot(x,m_estym,label='estymowany system')
plt.legend()
plt.grid()
'''

'''
# p4
a=1
x=np.linspace(-2,2,N)
Y=[m(X[i])+Z[i] for i in range(len(X))]

K=[K1,K2,K3,K4]
m_oryg=[m(i) for i in x]
plt.figure(figsize=(10,5))
plt.plot(x,m_oryg,label='system oryginalny')
jadra=['Jadro prostokatne','Jadro Gaussowskie','Jadro Epanechnikova','Jadro Tricube']


for k in range(len(K)):
    m_estym = [m_est(i, X, Y, hmin, K[k]) for i in x]
    plt.plot(x,m_estym, label='{}'.format(jadra[k]))
plt.legend()
#plt.title('a = ' + str(a)) 
plt.grid()
print(hmin)
'''


'''
a=1000
x=np.linspace(-2,2,N)
Y=[m(X[i])+Z[i] for i in range(len(X))]

hn=np.arange(0.05,1,0.01)
wek=[valid(X,Y,h,K1) for h in hn]
idx=wek.index(min(wek))
hmin=hn[idx]

K=[K1,K2,K3,K4]
m_oryg=[m(i) for i in x]
plt.figure(figsize=(10,5))

plt.plot(x,m_oryg,label='system oryginalny')
jadra=['Jadro prostokatne','Jadro Gaussowskie','Jadro Epanechnikova','Jadro Tricube']
for k in range(len(K)):
    m_estym = [m_est(i, X, Y, hmin, K[k]) for i in x]
    plt.plot(x,m_estym, label='{}'.format(jadra[k]))
plt.legend()
#plt.title('a = ' + str(a)) 
plt.grid()
print(hmin)
############################### zad 7 ###############################

'''