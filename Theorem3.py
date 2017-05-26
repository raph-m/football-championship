import Simulation_Championnat as sc
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import Theorem3_a_concrete_example
"""
This theorem enables to compute the value of the strength of an additionnal player in order for him to be able to win
when N grows to infinity.
"""

#the number of players
Ns = [10,20,50,70,85,100,120,140,160,180,200]
n = 600 #number of simulations
coef = 0.3

x_strong = np.zeros(len(Ns))
x_week = np.zeros(len(Ns))
i=0

for N in Ns:
    print("Computing N = "+str(N))
    x_strong[i] = sc.theorem3_bis(N,n,coef)
    x_week[i] = sc.theorem3_bis(N,n,-coef)
    print("chances of winning with a strenght of 1+(1+"+str(coef)+")*eps(N): "+str(x_strong[i])+"%")
    print("chances of winning with a strenght of 1+(1-"+str(coef)+")*eps(N): "+str(x_week[i])+"%")
    i+=1


plt.plot(Ns,x_strong,'r')
plt.plot(Ns,x_week)
plt.title("chances of winning for a strong team (red) vs a weak team (blue) when the number of players increases. With e ="+str(coef))
plt.xlabel("N the number of players")
plt.ylabel("% of chances of winning")
plt.show()
