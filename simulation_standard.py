"""Ce programme permet de simuler un grand nombre de championnats et permet d'évaluer la pobabilité de gagner pour
des équipes ayant une force suffisamment élévée. Attention, ici les matchs retours n'ont pas encore été implémentés"""
import Simulation_Championnat as sc
import numpy as np

#le nombre de championnats qu'on va simuler
n = 1000

print("according to our model, we compute the chances of winning for the best teams (given with a 95% interval)")
for name in ["Chelsea","ManCity","Arsenal","ManU","Tottenham"]:
    print("proba of winning for " + name + " with n = " + str(n) + ": " + str(100*sc.winning_championship_proba(sc.V,n,name))+"% +- "+str(196*sc.get_variance(sc.V,n,name)/np.sqrt(n))+"%")

print("")
print("We now compare it to the probabilities according to the bookmakers")
x = [8.0/(8+13),2.0/7,2./9,1./6,1./101]
name = ["Chelsea","ManCity","Arsenal","ManU","Tottenham"]
for i in range(len(x)):
    print(name[i]+": "+str(x[i]*100)+"%")


"""Une exemple de sortie: """

"""
proba of winning for Chelsea with n = 1000: 35.3%
proba of winning for ManCity with n = 1000: 26.6%
proba of winning for Arsenal with n = 1000: 21.2%
proba of winning for ManU with n = 1000: 14.799999999999999%
proba of winning for Tottenham with n = 1000: 9.0%

We now compare it to the probabilities according to the bookmakers
Chelsea: 38.095238095238095%
ManCity: 28.57142857142857%
Arsenal: 22.22222222222222%
ManU: 16.666666666666664%
Tottenham: 0.9900990099009901%

"""