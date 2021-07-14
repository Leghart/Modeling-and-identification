import matplotlib.pyplot as plt
import math 
import random as rand
import numpy as np

############################ zad 1 ###############################
def rand1(x0,z,N):
    x=[]
    x.append(x0)
  
    for i in range(1,N):
        x.append(z*x[-1]-math.floor(z*x[-1]))

    return x

def rand2(x0,N,c,m):
    k=N
    X=[]
    A=[]
    X.append(x0)
    for i in range(1,k-1):
    #A.append(b*X[-1]-math.floor(b*X[-1]))
        A.append(2*i+1)
        pom=X
        pom=pom[::-1]
        X.append((np.dot(A,pom)+c)%m)
    return X

#x0=0.42
z=5
N=1000


BINS=45
WIDTH=0.6



x=[rand1(0.242,999,N),rand1(0.241,31,N),rand1(0.241,111,N)]
'''
fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].hist(x,bins=BINS,rwidth=WIDTH)
axs[0].set_title("hist")
axs[1].plot(x,'.')
axs[1].set_title("plot")
'''

#plt.hist(x,bins=BINS,rwidth=WIDTH)

fig,axs=plt.subplots(1,3,figsize=(10,5))
axs[0].hist(x[0],bins=BINS,rwidth=WIDTH)
axs[0].set_title("X0=0.2")
axs[1].hist(x[1],bins=BINS,rwidth=WIDTH)
axs[1].set_title("X0=0.42")
axs[2].hist(x[2],bins=BINS,rwidth=WIDTH)
axs[2].set_title("X0=0.97")



###################### zad 2 ##########################################
    
'''
m=1
k=N
c=0
x0=0.2


x=[]

#for i in x0:
    #x.append(rand2(i,N,c,m))
    
x=rand2(x0, N, c, m)
okres=0
for i in x[1:]:
    if x0==i:
        print(okres)
    else:
        okres+=1
print(okres)

fig,axs=plt.subplots(1,3,figsize=(10,5))
axs[0].hist(x[0],bins=BINS,rwidth=WIDTH)
axs[0].set_title("X0=0.2")
axs[1].hist(x[1],bins=BINS,rwidth=WIDTH)
axs[1].set_title("X0=0.55")
axs[2].hist(x[2],bins=BINS,rwidth=WIDTH)
axs[2].set_title("X0=0.71")
'''

'''
for i in range(1,len(x)):
    for j in range(0,i):
        if x[i]==x[j] and flaga==0:
            okres=i
            flaga=1


'''