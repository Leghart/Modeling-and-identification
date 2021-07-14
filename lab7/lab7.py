import math
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from numpy import random
import time
from scipy.stats import cauchy


#################### zad 1 #################################################
a = 1
#m = lambda x: (0 <= abs(x) < 1) * (a*x**2) + (1 <= abs(x) < 2) * 1 + (2 <= abs(x) < 100000) * 0

def m(x):
    if 0 <= abs(x) and abs(x) < 1:
        return 1*x**2
    elif 1 <= abs(x) and abs(x) < 2:
        return 1
    elif 2 <= abs(x) and abs(x) < 10000:
        return 0

N = 500
x = np.linspace(-3,3,N)
m_org = [m(x) for x in x]

#plt.figure(figsize=(10,5))

#plt.grid()

X = random.uniform(-math.pi, math.pi, N)
sig = 1
Z = random.normal(0, sig, N)

###################### zad 2 ###############################################
Y = [m(X[i]) + Z[i] for i in range(len(X))]

'''
plt.figure(figsize=(10,5))
plt.plot(X, Y, '.',label='zaszumione pomiary')
plt.plot(x,m_org,label='system oryginalny')
plt.legend()
plt.grid()
'''

###################### zad 3 + 4 ###########################################
  
def phik(x,k):
    if k == 0:
        return np.sqrt(1/(2*math.pi)) 
    else:
        return np.sqrt(1/math.pi)*np.cos(k*x)

def ak(X,Y,k):
    return sum([Y[n]*phik(X[n],k) for n in range(len(X))])/len(X)

def bk(X,k):
    return sum([phik(X[n],k) for n in range(len(X))])/len(X)

def gN(x,X,Y,L):
    return sum([ak(X,Y,k)*phik(x,k) for k in range(L)])

def fN(x,X,Y,L):
    return sum([bk(X,k)*phik(x,k) for k in range(L)])
    

def m_est(x,L):
    if fN(x,X,Y,L) == 0:
        return 0
    else:
        return gN(x,X,Y,L)/fN(x,X,Y,L)
    
'''
x = np.linspace(-3,3,N)
L=[2,3,4]
fig,axs=plt.subplots(1,3,figsize=(15,5))

for l in range(len(L)):
    out = [m_est(i,L[l]) for i in x] 
    axs[l].plot(x,out, label='estymator')
    axs[l].plot(x, m_org, label = 'system oryginalny')
    axs[l].legend()
    axs[l].grid() 
    axs[l].set_title('L = {}'.format(L[l]))
'''

######################### zad 5 + zad 6 ###########################################

def valid(L):
    Q=100
    suma=sum([(m_est(2*q/Q,L)-m(2*q/Q))**2 for q in range(-Q,Q,1)])
    return suma/(2*Q)
  

'''
L = np.arange(0,10,1)
wek=[valid(l) for l in L]
plt.figure(figsize=(10,5))
plt.plot(L,wek)
plt.xlabel('L')
plt.ylabel('valid(L)')
idx=wek.index(min(wek))
plt.grid()
L_opt = L[idx]
print(L_opt)

    
x = np.linspace(-3,3,N)
L=L_opt
out = [m_est(i,L) for i in x] 
plt.figure(figsize=(10,5))
plt.plot(x,out, label='ortogonalny estymator')
plt.plot(x, m_org, label = 'system oryginalny')
plt.legend()
plt.grid() 

############### zad 7 ########################################
'''


x = np.linspace(-3,3,N)
#D = cauchy(0,0.01)
#Z = D.pdf(x)
m_org = [m(x) for x in x]

X = random.uniform(-math.pi, math.pi, N)
Z = np.random.standard_cauchy(N)
Y = [m(X[i]) + Z[i] for i in range(len(X))]

plt.figure(figsize=(10,5))
plt.plot(X, Y, '.',label='zaszumione pomiary')
plt.plot(x,m_org,label='system oryginalny')
plt.legend()
plt.grid()

L = np.arange(0,10,1)
wek=[valid(l) for l in L]
plt.figure(figsize=(10,5))
plt.plot(L,wek)
plt.xlabel('L')
plt.ylabel('valid(L)')
idx=wek.index(min(wek))
plt.grid()
L_opt = L[idx]
print(L_opt)

x = np.linspace(-3,3,N)
L= L_opt
out = [m_est(i,L) for i in x] 
plt.figure(figsize=(10,5))
plt.plot(x,out, label='ortogonalny estymator')
plt.plot(x, m_org, label = 'system oryginalny')
plt.legend()
plt.grid() 

# wniosek jesla dne sa niezapszumione to najlepiej jest uzyc l->oo, kiedy sa zaszumione
# a i b sa wyznaczane na podstawie Y czyli bledy beda sie nawarstiwac (duzo probek ma duze bledy)



L=[1,2,3]
fig,axs=plt.subplots(1,3,figsize=(15,5))

for l in range(len(L)):
    out = [m_est(i,L[l]) for i in x] 
    axs[l].plot(x,out, label='estymator')
    axs[l].plot(x, m_org, label = 'system oryginalny')
    axs[l].legend()
    axs[l].grid() 
    axs[l].set_title('L = {}'.format(L[l]))

'''
L=[9,11,15]
fig,axs=plt.subplots(1,3,figsize=(15,5))

for l in range(len(L)):
    out = [m_est(i,L[l]) for i in x] 
    axs[l].plot(x,out, label='estymator')
    axs[l].plot(x, m_org, label = 'system oryginalny')
    axs[l].legend()
    axs[l].grid() 
    axs[l].set_title('L = {}'.format(L[l]))
'''
'''
L=16
out = [m_est(i,L) for i in x] 
plt.figure(figsize=(10,5))
plt.plot(x, out, label='estymator')
plt.plot(x, m_org, label = 'system oryginalny')
plt.legend()
plt.grid() 
'''


