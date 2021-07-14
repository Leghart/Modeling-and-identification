import matplotlib.pyplot as plt
import random as rand
from numpy import random
import numpy as np
from math import sqrt,floor,e,log
import math


def rand1(a,b,N):
    x0=0.114212
    x=[]
    x.append(x0)
    z=111
     
    for i in range(1,N):
        x.append(z*x[-1]-math.floor(z*x[-1]))

    out=[]
    for i in range(len(x)):
        out.append(b*x[i]-a)
    return out


def znajdz_min(a,b,f):
    xtest=np.arange(a-1,b+1,0.01)
    
    roz=np.empty(0)
    for i in xtest:
        roz=np.append(roz,f(i))
    
    dmin=max(roz)       
    return dmin

################# PANEL STEROWANIA ###########################
N=100000

BINS=55
WIDTH=0.7


############################ zad 1 ###############################
'''
a=-1
b=1
f=(lambda x: (-1<x<=0)*(x+1)+(0<x<=1)*(-x+1))


d=1


u1=random.uniform(a,b,N)
u2=random.uniform(0,d,N)

X=np.empty(0)
for i in range(N):
    if u2[i]<=f(u1[i]):      
        X=np.append(X,u1[i])


d=5
u1=random.uniform(a,b,N)
u2=random.uniform(0,d,N)
X_por=np.empty(0)
for i in range(N):
    if u2[i]<=f(u1[i]):      
        X_por=np.append(X_por,u1[i])
        
print("========= zad1 ============ ")        
print("% Probki I: ",len(X)*100/N)
print("% Probki II: ",len(X_por)*100/N)    
    
fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].hist(X,bins=BINS,rwidth=WIDTH)
axs[0].set_title("d1=1")
axs[1].hist(X_por,bins=BINS,rwidth=WIDTH)
axs[1].set_title("d2=5")
'''
###################### zad 2 ##########################################
'''
c=100/198
a=0
b=1


f=(lambda x: (0<x<0.01)*(50)+(0.01<x<1)*(c))


d=znajdz_min(-1, 1, f)

u1=random.uniform(a,b,N)
u2=random.uniform(0,d,N)
X=np.empty(0)


for i in range(N):
    if u2[i]<=f(u1[i]):      
        X=np.append(X,u1[i])
print("========= zad2 ============ ")
print("% Probki I: ",len(X)*100/N)   
    
plt.figure(2)
plt.hist(X,bins=BINS,rwidth=WIDTH)
'''
############## inne podejscie ########################

u=random.rand(N)
v=random.rand(N)

g=lambda x:(1/2*math.exp(-abs(x)))
f=(lambda x: (0<x<0.01)*(50)+(0.01<x<1)*(100/198))

c=sqrt(2*e/math.pi)

znak=map(lambda k:(k<=1/2)*(1)+(k>1/2)*(-1),v)
Z=list(znak)
Fodw=map(lambda k,Z: -math.log(1-k,math.e)*Z,u,Z)
v=list(Fodw)

u=random.rand(N)
#u=np.random.normal(0,1,N)
tmp=[]
X=[]
for i in range(N):
    if u[i]*c*g(v[i])<=f(v[i]):
        tmp.append(v[i])   
       
for i in range(len(tmp)):
    zz=2*math.floor(2*random.rand())-1
    X.append(zz*tmp[i])

plt.hist(X,bins=BINS,rwidth=WIDTH)

###################### zad 3 ##########################################
'''
r=sqrt(2/math.pi)
Pkw=2*r**2
a=-r
b=r
d=r

f=lambda x: (a<=x<=b)*(np.sqrt(r**2-x**2))

u1=random.uniform(a,b,N)
u2=random.uniform(0,d,N)
X=np.empty(0)

il_prob=0
for i in range(N):
    if u2[i]<=f(u1[i]):      
        il_prob+=1
        X=np.append(X,u1[i])

pi=4*d*len(X)/(r*N)

print("========= zad3 ============ ")
print('Obliczone pi I: '+str(pi)) 
print("eta 1: ",len(X)*100/N)       


d=2
u1=random.uniform(a,b,N)
u2=random.uniform(0,d,N)
X_por=np.empty(0)

il_prob=0
for i in range(N):
    if u2[i]<=f(u1[i]):      
        il_prob+=1
        X_por=np.append(X_por,u1[i])



pi=4*d*len(X_por)/(r*N)
print('Obliczone pi II: '+str(pi))        
print("eta 2: ",len(X_por)*100/N)

fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].hist(X,bins=BINS,rwidth=WIDTH)
axs[0].set_title("d1=$\sqrt{2/\pi}$",)
axs[1].hist(X_por,bins=BINS,rwidth=WIDTH)
axs[1].set_title("d2=2")

###################### zad 4 ##########################################

u=random.rand(N)
v=random.rand(N)

g=lambda x:(1/2*math.exp(-abs(x)))
f=lambda x:e**(-x**2/2)/sqrt(2*math.pi)

c=sqrt(2*e/math.pi)

znak=map(lambda k:(k<=1/2)*(1)+(k>1/2)*(-1),v)
Z=list(znak)
Fodw=map(lambda k,Z: -math.log(1-k,math.e)*Z,u,Z)
v=list(Fodw)

u=random.rand(N)

tmp=[]
X=[]
for i in range(N):
    if u[i]*c*g(v[i])<=f(v[i]):
        tmp.append(v[i])   
       
for i in range(len(tmp)):
    zz=2*math.floor(2*random.rand())-1
    X.append(zz*tmp[i])
  
#######
c=5

znak=map(lambda k:(k<=1/2)*(1)+(k>1/2)*(-1),v)
Z=list(znak)
Fodw=map(lambda k,Z: -math.log(1-k,math.e)*Z,u,Z)
v=list(Fodw)

u=random.rand(N)

tmp=[]
X_por=[]
for i in range(N):
    if u[i]*c*g(v[i])<=f(v[i]):
        tmp.append(v[i])      
        
for i in range(len(tmp)):
    zz=2*math.floor(2*random.rand())-1
    X_por.append(zz*tmp[i])    
 
print("========= zad4 ============ ")    
print("% Probki I: ",len(X)*100/N)
print("% Probki II: ",len(X_por)*100/N)        
fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].hist(X,bins=BINS,rwidth=WIDTH)
axs[0].set_title("c1=$\sqrt{2e/\pi}$")
axs[1].hist(X_por,bins=BINS,rwidth=WIDTH)
axs[1].set_title("c2=5")


########################## zad dod #############################
'''

def pnorm(mu = 0, sigma = 1,N=100000):
    rozklad=[]
    for i in range(int(N/2)):
        u1=random.rand()
        u2=random.rand()        
        pierw=sigma*sqrt(-2*log(u1))
        rozklad.append(pierw*math.cos(2*math.pi*u2)+mu)
        rozklad.append(pierw*math.sin(2*math.pi*u2)+mu)
    return rozklad

mu=2
sig=1
x_moj = pnorm(mu,sig,N)
x_rand=np.random.normal(mu,sig,N)

fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].hist(x_moj,bins=BINS,rwidth=WIDTH)
axs[0].set_title("Box-Muller")
axs[1].hist(x_rand,bins=BINS,rwidth=WIDTH)
axs[1].set_title("numpy")

'''
################## zad dod 2 ###############
'''
def gen_roz(fun,N,a,b):
    x=np.arange(a,b,0.01) # wektor próbek nosnika
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


f=lambda x: (-1<x<=0)*(x+1)+(0<=x<1)*(-x+1)
h=lambda x: (0<x<=0.6)*(x)+(0.6<x<=2)*(2-x)
#r=1
#g=f=lambda x: (-r<=x<=r)*(np.sqrt(r**2-x**2))

a=gen_roz(h,10000,0,2)
#plt.hist(a,bins=BINS,rwidth=WIDTH)

def est1(T):
    return sum(T)/len(T)

print(est1(a))
'''
'''
xx=np.arange(0,2,0.001)
yy=np.empty(len(xx))
for i in range(len(xx)):
    yy[i]=h(xx[i])

fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].hist(a,bins=BINS,rwidth=WIDTH)
axs[0].set_title("Histogram")
axs[1].plot(xx,yy)
axs[1].set_title("Funkcja f(x)")
'''
