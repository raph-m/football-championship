"""La méthode d'échantillonage préférentiel permet de calculer la probabilité des évènement rares avec moins de simulations:"""

import Simulation_Championnat as sc

n=100

print("Computing the proba of Leicester winning with standard simulation: (n = "+str(n)+")")
print(sc.winning_championship_proba(sc.V,n,"Leicester"))
print("")

print("Computing the proba of Leicester winning with preferential simulation: (n = "+str(n)+")")
print(sc.rare_event_basic("Leicester",n))
print("")

print("Computing the proba of Norwich winning with preferential simulation: (n = "+str(n)+")")
print(sc.rare_event_basic("Norwich",n))
print("")

n=10*n

print("Computing the proba of the very rare event that occured during the 2015 Premier League with preferential simulation: (n = "+str(n)+")")
print(sc.rare_event_complex(n))
print("")
