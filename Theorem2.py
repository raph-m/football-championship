""""illustration du théorème 2:
On sait que l'assumption A est respectée pour différentes distributions:
*la distribution uniforme avec alpha = 1
*la distribution arcsin avec alpha = 1/2
*les distribution Beta B(a,b) avec alpha = b (et b<2)

Le théorème 2 montre alors que:
*pour tout gamma < 1 - alpha/2,
les N**gamma meileurs joueurs ne gagnent presque jamais

*pour tout gamma > 1 - alpha/2,
les N**gamma meilleurs joueurs gagnent presque toujours
    """
import Simulation_Championnat as sc


""""
On peut illuster le théorème en faisant des essais pour différentes valeurs du nombre de joueurs et vérifier que la
portion de joueurs N**gamma finit toujours par gagner (respectivement ne pas gagner)
On trace donc le graphe de en fonction de gamma pour des valeurs de N allant vers l'infini
Les essais sont stockés dans le dossier résultats. Ils ont été faits sans matchs retour. Pour avoir une meilleure
illustration  du théorème, on pourra envisager de faire des calculs pour de plus grandes valeurs de N.
Dans le graphes, on note la présence d'une barre verticale qui montre le rupture en entre les gamma qui vont tendre
vers 0 et ceux qui vont tendre vers 1
"""

#the method works with law = uniform or law = beta and you must then specify the values of a and b
law = "uniform"
a=1.
b=1.
n=500 #number of championships simulated

Ns = [10,20,50,100,150,200]
for N in Ns:
    print("Computing with N = "+str(N))
    sc.theorem_2(N, n, "uniform", a, b)

