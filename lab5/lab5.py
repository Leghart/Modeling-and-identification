import matplotlib.pyplot as plt
import random as rand
from numpy import random
from functools import reduce
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



################# PANEL STEROWANIA ###########################

BINS=55
WIDTH=0.7

############################ zad 1 ###############################
N=1000
mu=1
sig=1
#X=random.normal(mu,sig,N)
#plt.hist(X,bins=BINS,rwidth=WIDTH)

############################ zad 2 ###############################

#I=lambda x: 1*(abs(x)<=1)+0*(abs(x)>1)

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
 
 
N = 2000    
X=random.normal(1,1,N)

x=np.linspace(-3,5,N) 
wek_gest=[f(mu,sig,i) for i in x]
wek_est1=[est_jadr(X, i, 0.5,K1) for i in x]
    
plt.plot(x,wek_est1,label="estymator")
plt.plot(x,wek_gest,label='gestosc')
plt.grid(which='both')
plt.legend()
#axs[a].set_title('$h_N$ = '+str(hN[a]))
'''
t_start=time.time()
Ilosc=2000
x=np.linspace(-3,5,N) 

hN=[0.1,0.45,2]
fig,axs=plt.subplots(1,3,figsize=(20,8))
for a in range(len(hN)):

    wek_gest=[f(mu,sig,i) for i in x]
    wek_est1=[est_jadr(X, i, hN[a],K1) for i in x]
    
    axs[a].plot(x,wek_est1,label="estymator")
    axs[a].plot(x,wek_gest,label='gestosc')
    axs[a].grid(which='both')
    axs[a].legend()
    axs[a].set_title('$h_N$ = '+str(hN[a]))

t_stop=time.time()
print('Czas trwania:'+str(t_stop-t_start))

############################ zad 3 ###############################

######################## dla rozkladu normalnego #################

X=np.random.normal(mu,sig,N)
Ilosc=1000
x=np.linspace(-3,5,Ilosc)

hN=0.5

wek_gest=[(f(mu,sig,i)) for i in x]
wek_est1=[(est_jadr(X, i, hN,K1)) for i in x]
wek_est2=[(est_jadr(X, i, hN,K2)) for i in x]
wek_est3=[(est_jadr(X, i, hN,K3)) for i in x]  
wek_est4=[(est_jadr(X, i, hN,K4)) for i in x]    

plt.figure(figsize=(15,5))
plt.plot(x,wek_est1,label="jadro prostokatne")
plt.plot(x,wek_est2,label="jadro Gaussowskie")
plt.plot(x,wek_est3,label="jadro Epanechnikova")
plt.plot(x,wek_est4,label="jadro Tricube")
plt.plot(x,wek_gest,label="funkcja gestosci")
plt.grid(True,which='both')
plt.legend()
'''

######################## dla rozkladu jednostajengo #################
'''
#g=lambda x: (-1<x<=0)*(x+1)+(0<x<=1)*(-x+1)
a=0
b=1

#g=lambda x: (a<=x<=b)*1
c=100/198
g=(lambda x: (0<x<0.01)*(50)+(0.01<x<1)*(c))

N=2000
X=gen_roz(g, N, a, b)

Ilosc=2000
#x=np.linspace(0,0.5,Ilosc)
x=np.arange(0,0.2,1/Ilosc)


hN=0.0009
wek_gest=[(g(i)) for i in x]
wek_est1=[(est_jadr(X, i, hN,K1)) for i in x]
wek_est2=[(est_jadr(X, i, hN,K2)) for i in x]
wek_est3=[(est_jadr(X, i, hN,K3)) for i in x]  
wek_est4=[(est_jadr(X, i, hN,K4)) for i in x]     

plt.figure(figsize=(15,5))
plt.plot(x,wek_est1,label="jadro prostokatne")
plt.plot(x,wek_est2,label="jadro Gaussowskie")
plt.plot(x,wek_est3,label="jadro Epanechnikova")
plt.plot(x,wek_est4,label="jadro Tricube")
plt.plot(x,wek_gest,label="funkcja gestosci")
plt.grid(True,which='both')
plt.legend()


'''

'''
def Err(x,hN,N):
    L=10
    M=N    
    gest=([f(mu,sig,i) for i in x]) 
    X=np.random.normal(mu,sig,N)
    SS=0    
    for _ in range(L):       
        fn=list(map(lambda v:est_jadr(X, v, hN, K1),x))
        s=[(fn[k]-gest[k])**2 for k in range(M)]
        SS+=np.sum(np.array(s))
    return SS/(L*M)
 

N=100
h = np.arange(0.01,20,0.05)
przedzial = np.linspace(-2,4,N)


tstart=time.time()
bl=[Err(przedzial,i,N) for i in h]    
tstop=time.time()
plt.figure(figsize=(10,5))
plt.plot(h,bl)  
plt.grid(which='both') 
         
print("czas: "+str(tstop-tstart))

'''

######################### Wnioski #######################
# piki powstaja dlatego ze estymator wpada w ekstrema lokalne, 
# gdy rozklady sa pojebane szybkie uzycie malego h jest konieczne     

# dla szybkozmiennych prostkatny jest kozak
# dla gladkich rozkladow lepszy jest gaussowski

############################ zad 4 ###############################
