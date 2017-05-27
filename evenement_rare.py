"""La méthode d'échantillonage préférentiel permet de calculer la probabilité des évènement rares avec moins de simulations:"""

"""
Il y a une simplification des calculs possible quand on parle des évènements simples comme "Leicester gagne".
Effectivement, pour provoquer cet évènement, on va simplement augmenter la force de Leicester et ainsi les
expériences de Bernouilli entre deux équipes qui ne sont pas Leicester resteront inchangé.(ce qui donne dans la formule
des termes qui valent systématiquement 1.
Cependant, pour des évènements plus complexes comme l'évènement complexe défini dans le sujet, augmenter la force de
Leicester ne suffit pas à provoquer l'évènement. Il faut donc aussi diminuer les forces des équipes concernées (Liverpool,
Chelsea, etc). La simplification n'est alors plus possible...
On peut noter que malgré tout les décalages en probabilité, l'évènement est extrêment rare et que pour obtenir un
intervalle de confiance correcte il faut faire de très nombreux essais quand même
"""


import Simulation_Championnat as sc
import numpy as np
n=10000
name = "Leicester"

print("Computing the proba of Leicester winning with standard simulation: (n = "+str(n)+")")
print(str(sc.winning_championship_proba(sc.V,n,"Leicester")*100)+"%")
print("")


print("Computing the proba of "+name+" winning with preferential simulation: (n = "+str(n)+")")
print(str(sc.rare_event_basic(name,n)*100)+"%  +-"+str(196*sc.get_variance_rare_event_basic(name,n)/np.sqrt(n))+"%")
print("")

name = "Sunderland"
print("Computing the proba of "+name+" winning with preferential simulation: (n = "+str(n)+")")
print(str(sc.rare_event_basic(name,n)*100)+"%  +-"+str(196*sc.get_variance_rare_event_basic(name,n)/np.sqrt(n))+"%")
print("")

n=10*n

print("Computing the proba of the very rare event that occured during the 2015 Premier League with preferential simulation: (n = "+str(n)+")")
print(str(sc.rare_event_complex(n)*100)+"% +- "+ str(196*sc.get_variance_rare_event_basic(name,n)/np.sqrt(n))+"%")
print("")
