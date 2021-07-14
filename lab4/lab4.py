import matplotlib.pyplot as plt
import random as rand
from numpy import random
import numpy as np
import math


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

def pnorm(mu = 0, sigma = 1,N=100000):
    rozklad=[]
    for i in range(int(N/2)):
        u1=random.rand()
        u2=random.rand()        
        pierw=sigma*math.sqrt(-2*math.log(u1))
        rozklad.append(pierw*math.cos(2*math.pi*u2)+mu)
        rozklad.append(pierw*math.sin(2*math.pi*u2)+mu)
    return rozklad



################# PANEL STEROWANIA ###########################

BINS=55
WIDTH=0.7

############################ zad 1 ###############################

f=lambda x: (x<0)*(0)+(0<=x<=1)*(2*x)+(x>1)*1
#N=10000
#X=gen_roz(f,N,0,1)
#plt.hist(X,bins=BINS,rwidth=WIDTH)


########################### zad 2 ##################################

#dystrybuanta rozkladu z zad 1
F=lambda x: (0<=x<=1)*(x**2)+(x>1)*1 


# Dystrybuanta empiryczna
def Fe(X,x):
    pom=[]
    for i in range(len(X)):
        if X[i]<=x:
            pom.append(1)
    return sum(pom)/len(X)


'''
AA=[10,100,1000]
fig,axs=plt.subplots(1,3,figsize=(15,5))
for n in range(len(AA)):
    b=np.linspace(-1,1.5,AA[n])
    FE=[]
    FF=[]
    for i in b:
        FE.append(Fe(X,i))   
        FF.append(F(i))    
    
    axs[n].step(b,FE,label='Dystrybuanta empiryczna')
    axs[n].plot(b,FF,label='Dystrybuanta rozkładu')
    axs[n].grid(which='both')
    axs[n].legend()
    axs[n].set_title('N='+str(AA[n]))
'''

####################### zad 3 #######################################

def DN(X1,X2):
    return max(abs(np.array(X1)-np.array(X2)))
'''
AA=range(1,1000)
OUT=[]
D=[]


for n in range(len(AA)):
    X=gen_roz(f,AA[n],0,1)
    b=np.linspace(-1,1.5,AA[n])
    
    F=map(lambda x: (0<=x<=1)*(x**2)+(x>1)*1,b )  
    FF=list(F) 
    FE=[]
    for i in range(AA[n]):   
        FE.append(Fe(X,b[i]))

    OUT.append(DN(FF,FE))
  
#plt.figure(figsize=(15,5))
plt.plot(OUT)
plt.grid(which='both')
plt.xlabel('N')
plt.ylabel('$D_N$')

'''
############################ zad 4 ###############################
def dys_nor(mu,sig,x):
    return 1/2*(1+math.erf((x-mu)/(sig*math.sqrt(2))))
    
def dys_cau(x0,y,x):
    return 1/math.pi*np.arctan((x-x0)/y)+1/2


plik=open('ModelowanieLab4Data.txt','r')
data=plik.read().split('\n')
del data[-1]
X=np.array(data,dtype='f')

'''
N=1000
FE=[]
y_n1=[]
y_n2=[]
y_c=[]

b=np.linspace(-5,5,N)
 
for i in b:
    FE.append(Fe(X,i))   
    y_n1.append(dys_nor(-1,1,i))
    y_n2.append(dys_nor(0, 5, i))
    y_c.append(dys_cau(0,1,i))
    

# porownanie za pomoca D_N
print(DN(y_n1,FE))
print(DN(y_n2,FE))
print(DN(y_c,FE))

plt.figure(figsize=(15,5))
plt.step(b,FE,label='Dystrybuanta empiryczna')
plt.plot(b,y_n1,label='Rozklad N(-1,1)')
plt.plot(b,y_n2,label='Rozklad N(0,5)')
plt.plot(b,y_c,label='Rozklad C(0,1)')
plt.grid(which='both')
plt.legend()
'''

##################### zad 5 ############################

def est3(T,u):
    suma=0
    for i in T:
        suma+=(i-u)**2
    return (suma/(len(T)-1))


def est1(T):
    return sum(T)/len(T)

def est2(T,u):
    suma=0
    for i in T:
        suma+=(i-u)**2
    return (suma/(len(T)))

def Var(F_emp):
    out=[]
    for i in range(len(F_emp)):
        out.append((F_emp[i]*(1-F_emp[i]))/len(F_emp))
    return out

N=300

# Wzor
b=np.linspace(-1,2,N)
X=gen_roz(f,N,0,1)
FE=[]
FF=[]
for i in b:
    FE.append(Fe(X,i))
    FF.append(F(i))

V=Var(FE)


plt.figure(figsize=(10,5))
plt.plot(b,V,label='wzor')
plt.grid(which='both')
plt.xlabel('F(x)')
plt.ylabel('$Var(\hat{F}(x)$')

'''
OUT=[]
b=np.linspace(-1,2,N)
# Macierz poszczegolnych realizacji
for i in range(N):
    FE=[]  
    XX=gen_roz(f,N,0,1)
    for i in b:
        FE.append(Fe(XX,i))
    OUT.append(FE)


TOTALNY_OUT=[]
for a in range(N):
    WEK=[]
    for i in range(N):
        WEK.append(OUT[i][a])

    TOTALNY_OUT.append(est3(WEK,est1(WEK)))


plt.plot(b,TOTALNY_OUT,label='eksperyment')
plt.legend()
'''