import matplotlib.pyplot as plt
import random as rand
from numpy import random
import numpy as np
from math import sqrt,floor,e,log
import math


def rand1(x0,z,N):
    x=[]
    x.append(x0)
    for i in range(1,N):
        x.append(z*x[-1]-math.floor(z*x[-1]))
    return x

def pnorm(mu = 0, sigma = 1,N=100000):
    rozklad=[]
    for i in range(int(N/2)):
        u1=random.rand()
        u2=random.rand()        
        pierw=sigma*sqrt(-2*log(u1))
        rozklad.append(pierw*math.cos(2*math.pi*u2)+mu)
        rozklad.append(pierw*math.sin(2*math.pi*u2)+mu)
    return rozklad  

################# PANEL STEROWANIA ###########################
N=1_000_000

BINS=55
WIDTH=0.7

############################ zad 1 ###############################
mu=10
sig=40
#T=np.random.normal(mu,sig,N)
#tmp=pnorm(mu,sig,N)
#T_wl = np.array(tmp)

# nieobciazony estymator wartosci oczekiwanej
def est1(T):
    return sum(T)/len(T)

# obciazony estymator wariancji
def est2(T,u):
    return sqrt(np.sum((T-u)**2)/len(T))

# nieobciazony estymator wariancji
def est3(T,u):
    return sqrt(np.sum((T-u)**2)/(len(T)-1))



'''
W=[10,100,1000,10000,100000]
for i in W:
    N=i
    out=0
    out1=0
    sn=0
    sn1=0
    for ile in range(200):
        T=np.random.normal(mu,sig,i)
        tmp=pnorm(mu,sig,i)
        T_wl = np.array(tmp)    
        out=est1(T)
        out1=est1(T_wl)
        sn=sn+est3(T,out)
        sn1=sn1+est3(T_wl,out1)
    A=sn/200
    B=sn1/200
    print("Gotowy: "+str(round(A,4))+" "+"MOJ: "+str(round(B,4)))

T=np.random.normal(mu,sig,N)
tmp=pnorm(mu,sig,N)
T_wl = np.array(tmp)  

out=est1(T)
out1=est1(T_wl)
sn=est2(T,out)
sn1=est2(T_wl,out1)
print(str(sn))

Sn=est3(T,out)
Sn1=est3(T_wl,out1)
print(str(Sn))
'''
############################ zad 2 ###############################

def Err(mi,N,L):
    suma=0
    for i in range(L):
        T=np.random.normal(mi,1,N)
        u_l=est1(T)
        suma=suma+(u_l-mi)**2   
    return suma/L

def Err_s(s,N,L):
    suma=0
    for i in range(L):
        T=np.random.normal(0,s**2,N)
        u=est1(T)
        s_e=est2(T,u)
        suma=suma+(s_e-s**2)**2   
    return suma/L

def Err_S(S,N,L):
    suma=0
    for i in range(L):
        T=np.random.normal(0,S**2,N)
        u=est1(T)
        S_e=est3(T,u)
        suma=suma+(S_e-S**2)**2   
    return suma/L




'''
def Err(uN,mu):
    suma=0
    for i in range(len(uN)):
        suma+=(uN[i]-mu)**2
    return suma/len(uN)

L=10
N=100
mu=0

uN=[]
for i in range(L):
    T=np.random.normal(0,1,N)
    uN.append(est1(T))


out1=[]
for n in range(1,N):
    uN=[]
    for i in range(L):
        T=np.random.normal(0,1,n)
        uN.append(est1(T))
    out1.append(Err(uN,mu))

L=100
out2=[]
for n in range(1,N):
    uN=[]
    for i in range(L):
        T=np.random.normal(0,1,n)
        uN.append(est1(T))
    out2.append(Err(uN,mu))


plt.plot(out1,label='L=10')
plt.plot(out2,label='L=10')
'''


#####################################################################
out1=[]
out2=[]
out3=[]
out4=[]
mi=2
sig=1

############# Porównanie estymatora wartosci oczekiwanej (zmiana L)#############
'''
L1=10
L2=100
for n in range(1,100):
    out1.append(Err(mi,n,L1))
    out2.append(Err(mi,n,L2))
mi=100
for n in range(1,100):
    out3.append(Err(mi,n,L1))
    out4.append(Err(mi,n,L2))

fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].plot(out1,label='L=10')
axs[0].plot(out2,label='L=100')
axs[0].set_xlabel('N')
axs[0].set_ylabel('Err(N)')
axs[0].set_title("$\mu=2$")
axs[0].grid(which='both')
axs[1].plot(out3,label='L=10')
axs[1].plot(out4,label='L=100')
axs[1].set_xlabel('N')
axs[1].set_ylabel('Err(N)')
axs[1].set_title("$\mu=10$")
axs[1].grid(which='both')
axs[0].legend()
axs[1].legend()
'''
############# Porównanie estymatora wariancji (obc) (zmiana L)#############
'''  
L1=10
L2=100
sig=2
N=100
for n in range(1,N):
    out1.append(Err_s(sig,n,L1))
    out2.append(Err_s(sig,n,L2))
sig=10
for n in range(1,N):
    out3.append(Err_s(sig,n,L1))
    out4.append(Err_s(sig,n,L2))
 
  
fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].plot(out1,label='L=10')
axs[0].plot(out2,label='L=100')
axs[0].set_xlabel('N')
axs[0].set_ylabel('Err(N)')
axs[0].set_title("$\sigma^2=2$")
axs[0].grid(which='both')
axs[1].plot(out3,label='L=10')
axs[1].plot(out4,label='L=100')
axs[1].set_xlabel('N')
axs[1].set_ylabel('Err(N)')
axs[1].set_title("$\sigma^2=10$")
axs[1].grid(which='both')
axs[0].legend()
axs[1].legend()
'''

############# Porównanie estymatora wariancji (nieobc)(zmiana L)#############
'''   
N=100
L1=10
L2=100
sig=3
for n in range(1,N):
    out1.append(Err_S(sig,n,L1))
    out2.append(Err_S(sig,n,L2))
sig=10
for n in range(1,N):
    out3.append(Err_S(sig,n,L1))
    out4.append(Err_S(sig,n,L2))



fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].plot(out1,label='L=10')
axs[0].plot(out2,label='L=100')
axs[0].set_xlabel('N')
axs[0].set_ylabel('Err(N)')
axs[0].set_title("$\sigma^2=2$")
axs[0].grid(which='both')
axs[1].plot(out3,label='L=10')
axs[1].plot(out4,label='L=100')
axs[1].set_xlabel('N')
axs[1].set_ylabel('Err(N)')
axs[1].set_title("$\sigma^2=10$")
axs[1].grid(which='both')
axs[0].legend()
axs[1].legend()
'''


########################### zad dod ###########################
'''
L=100
sig=2
mu=1
N=1000
for n in range(1,N):
    T=np.random.normal(mu,sig,N)
    out1.append(est2(T,mu))
    out2.append(est3(T,mu))

OUT=[]
for i in range(len(out1)):
    OUT.append(abs(out1[i]-out2[i]))
    
plt.figure()
plt.plot(OUT)
plt.xlabel("N")
plt.ylabel("$|\hat{s}^2-\hat{S}^2|$")

plt.grid(which='both')
'''
############################ zad 3 ###############################
'''
#rozklad cauchego
T = np.random.standard_cauchy(N)

#a=est1(T)
#print(est1(T))
#print(est2(T,a))


W=[10,100,1000,10000,100000]

for i in W:
    N=i
    out=0
    out1=0
    sn=0
    sn1=0
    for ile in range(200):
        T = np.random.standard_cauchy(N)
        X=np.random.normal(0,1,N)
        Y=np.random.normal(0,1,N)
        T_wl=X/Y
        out=est1(T)
        out1=est1(T_wl)
        sn=sn+est3(T,out)
        sn1=sn1+est3(T_wl,out1)

    A=sn/200
    B=sn1/200
    print("Gotowy: "+str(round(A,4))+" "+"MOJ: "+str(round(B,4)))

plt.hist(T)

'''

