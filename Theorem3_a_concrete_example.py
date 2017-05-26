import Simulation_Championnat as sc
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

""""Cet exemple reprend une situation qu'on peut retrouver dans la réalité (on peut penser au championnat français).
On compare deux situations:
*Une équipe possède une force de 2 tandis que toutes les autres possèdent une force de 1
*Une équipe possède une force de 2, une autre une force de 1 et tout le reste a 0.5
L'écart entre la force de l'équipe la plus forte et la moyenne est bien plus grand dans le deuxième cas. On s'attend
donc à obtenir de meilleurs résultats dans le cas deux.

Quel est le lien avec le théorème 3 ? On prend deux championnats: le schampionnats un et deux privé du meilleur joueur.
On se demande alors quelle force il faut avoir pour qu'un joueur additionnel puisse avoir des chances de gagner.

Remarquons d'abord que nos deux distributions respectent l'assomption A avec alpha = 0.
Ensuite, un calcul simple montre que nuU1>nuU2 puis que epsU2 > epsU1 pour tout N. Donc cela signifie, d'après le
théorème 3 que la force nécessaire à un joueur supplémentaire pour avoir des bonnes chances de gagner le tournoi est
supérieure dans le cas numéro 2. Ceci peut paraître contre-intuitif, mais il s'agit d'un phénomène sensible en fait.
Si toutes les équipes sont très faibles, le

 """""

def get_first_strength_uniform(N,n):
    ans = 0.
    strength = 0.45
    while(ans < 70):
        strength += 0.05
        print(strength)
        ans , poubelle = sc.theorem3(N,n,strength)
    print("strength uniform: "+str(strength))
    return strength

def get_first_strength_disparate(N,n):
    ans = 0.
    strength = 0.45
    while(ans < 70):
        strength += 0.05
        print(strength)
        poubelle, ans = sc.theorem3(N,n,strength)
    print("strength disparate: "+str(strength))
    return strength


def get_first_strength(uniform,N,n):
    if(uniform):
        return get_first_strength_uniform(N,n)
    else:
        return get_first_strength_disparate(N,n)


n=100
Ns = [20,30,40,50,60,70,85,100,120,130,140,160]
disparate = np.zeros(len(Ns))
uniform = np.zeros(len(Ns))
i=0

for N in Ns:
    print("Computing N = "+str(N))
    uniform[i] = get_first_strength(True,N,n)
    disparate[i] = get_first_strength(False,N,n)
    i+=1

plt.title("illustration of theorem 3 with "+str(n)+" simulations")
plt.xlabel("number of players")
plt.ylabel("strength needed for the additional player in order to win in 70% cases (red = disparate teams, blue = uniform teams")
plt.plot(Ns,uniform)
plt.plot(Ns,disparate,'r')
plt.show()
















