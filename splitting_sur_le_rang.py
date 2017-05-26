import Simulation_Championnat as sc
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


n_total = 100000
team_number = sc.get_number_with_name("Leicester")
bound_inf = np.arange(14,-1,-2)
print(bound_inf)
bound_sup = bound_inf + 2

z = len(bound_inf)
probas = np.zeros(z)
n=int(n_total/(z+1))

first_proba ,result = sc.proba_score2(bound_sup[0],team_number,sc.V,n)
print("first prob: "+str(100*first_proba)+"%")
ro = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.006,0.006,0.006,0.006,0.006,0.5,0.5,0.5]

for i in range(z):
    print("")
    print(i)
    print("ro = "+str(ro[i]))
    probas[i],result = sc.conditionnal_proba2(result,sc.V,ro[i],team_number,n,bound_inf[i],bound_sup[i])
    print("proba: "+str(100*probas[i])+"%")

print("Probabilité finale: "+str(100*np.prod(probas)*first_proba)+"%")





































