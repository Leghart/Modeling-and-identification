import matplotlib.pyplot as plt
import math 
import random as rand
from numpy import random
import numpy as np

##################### STARE GENERATORY ###################################
def rand1(N):
    #x0=rand.random()
    x0=0.17
    x=[]
    x.append(x0)
    z=3.2
     
    for i in range(1,N):
        x.append(z*x[-1]-math.floor(z*x[-1]))

    return x

def rand2(N):
    c=31
    m=1
    A=[]
    X=[]
    x0=rand.random()
    X.append(x0)
    for i in range(1,N):
        A.append(2*i+1)
        pom=X
        pom=pom[::-1]
        X.append((np.dot(A,pom)+c)%m)
    return X

################# PANEL STEROWANIA ###########################
N=100000
x = random.rand(N)
x1=rand1(N)
#x2=rand2(N)
Q1=x1
Q2=x

BINS=55
WIDTH=0.7
'''

############################ zad 1 ###############################
Fodw=map(lambda u: math.sqrt(u),Q1)
y11=list(Fodw)
Fodw=map(lambda u: math.sqrt(u),Q2)
y12=list(Fodw)

fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].hist(y11,bins=BINS,rwidth=WIDTH)
axs[0].set_title("generator piłokształtny")
axs[1].hist(y12,bins=BINS,rwidth=WIDTH)
axs[1].set_title("generator jednostajny")


###################### zad 2 ##########################################
Fodw=map(lambda q: (q<=1/2)*(math.sqrt(2*q)-1)+(q>1/2)*(1-math.sqrt(2-2*q)),Q1)
y21=(list(Fodw))
Fodw=map(lambda q: (q<=1/2)*(math.sqrt(2*q)-1)+(q>1/2)*(1-math.sqrt(2-2*q)),Q2)
y22=(list(Fodw))

fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].hist(y21,bins=BINS,rwidth=WIDTH)
axs[0].set_title("generator piłokształtny")
axs[1].hist(y22,bins=BINS,rwidth=WIDTH)
axs[1].set_title("generator jednostajny")

'''
###################### zad 3 ##########################################
Fodw=map(lambda q: (0<=q<1)*(-math.log(1-q,math.e)),Q1)
y31=(list(Fodw))
Fodw=map(lambda q: (0<=q<1)*(-math.log(1-q,math.e)),Q2)
y32=(list(Fodw))

fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].hist(y31,bins=BINS,rwidth=WIDTH)
axs[0].set_title("generator piłokształtny")
axs[1].hist(y32,bins=BINS,rwidth=WIDTH)
axs[1].set_title("generator jednostajny")

'''
###################### zad 4 ##########################################
v=random.rand(N)
znak=map(lambda k:(k<=1/2)*(1)+(k>1/2)*(-1),v)
Z=list(znak)
Fodw=map(lambda k,z: -math.log(1-k,math.e)*z,Q1,Z)
y41=list(Fodw)
Fodw=map(lambda k,z: -math.log(1-k,math.e)*z,Q2,Z)
y42=list(Fodw)

fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].hist(y41,bins=BINS,rwidth=WIDTH)
axs[0].set_title("generator piłokształtny")
axs[1].hist(y42,bins=BINS,rwidth=WIDTH)
axs[1].set_title("generator jednostajny")
'''