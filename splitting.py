import Simulation_Championnat as sc
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

""""Cette partie utilise les méthodes du théorème ergodique + splitting pour simuler la probabilité que Leicester gagne
Au lieu d'utiliser une part d'histoire de façon continue, on le fait de façon discrète.
En fait on part d'un résultat de championnat, et à partir de ce résultat nous allons resimuler une fraction ro des matchs
de ce championnat. Ainsi on construit bien une chaîne de Markov avec un nombre d'états finis. On respecte bien la propriété
de Markov car les probabilités de l'état n+1 ne dépendent que de l'état n. Le splitting se fait sur le score obtenu par
Leicester. On apllique ensuite le théorème ergodique sur chacun des morceaux, ce qui permet de récupérer les probas
conditionnelles puis les probas finales.

Leicester ne peut pas gagner avec moins de 10 points donc l'évènement Leicester gagne est bien
inclus dans "score(Leicester)>=10".
L'objectif est bien sûr de se rapprocher de cet évènement "score(Leicester)>=10" via des probas conditionnelles qui
doivent être les moins rares possibles.
On simule ici 190 matchs à chaque fois. Donc quand on verra ro = 0.006 cela signifie en fait qu'on ne resimule qu'un seul
match. Le problème est que le taux de rejection "idéal" pour les méthodes de splitting est d'environ 20%. Cependant,
admettons que l'on se place dans la situation ou Leicester a 9 points. Ses meilleures chances de gagner sont lorsque l'on
ne resimule qu'un seul match. Cependant même dans ce cas la, comme Leicester a des scores faibles par rapport aux autres
équipes, le taux de rejection sera très probablement élevé. C'est effectivement ce que l'on constate en pratique avec
des taux autour de 60%, voire 85% pour des scores de 10 points.
La méthode de splitting n'est donc pas forcément idéale dans ce cas.

Une autre approche que l'on pourrait développer est de faire du splitting sur le classement de Leicester. Ainsi les
évènements s'emboîtent mieux. Cependant, on pourrait être confronté à des problèmes car on ne sait pas très bien l'effet
des resimulations de quelques matchs sur le classement.
"""

n_total = 10000
team_number = sc.get_number_with_name("Leicester")
bound_inf = np.arange(5,11,1)
print(bound_inf)
bound_sup = bound_inf + 1

z = len(bound_inf)
probas = np.zeros(z)
n=int(n_total/(2*z+1))

first_proba ,result = sc.proba_score(10,team_number,sc.V,n)
print("first prob: "+str(100*first_proba)+"%")
ro = [0.5, 0.4, 0.1, 0.01, 0.006, 1., 0.006, 0.006]
for i in range(z):
    print("")
    print(i)
    if (i<z-1):
        probas[i],result = sc.conditionnal_proba(result,sc.V,ro[i],team_number,n,bound_inf[i],bound_sup[i])
        print("proba: "+str(100*probas[i])+"%")
    else:
        probas[i],result = sc.conditionnal_proba(result,sc.V,ro[i],team_number,10*n,bound_inf[i],last_bound=True)
        print("ro = "+str(ro[i]))
        print("last prob: "+str(probas[i]*100)+"%")

print("Probabilité finale: "+str(100*np.prod(probas)*first_proba)+"%")





































