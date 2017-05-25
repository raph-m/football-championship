import Simulation_Championnat as sc
import numpy as np
import matplotlib.pyplot as plt


"""Objectif: On fait l'hypothèse suivante: on souhaite se faire de l'argent à l'aide d'arbitrage sur des cotes de
bookmakers. On se pose comme modèle le modèle avec kappa ou teta. Nous partons du principe que les bookmakers sont
précis sur les évènements assez fréquent comme la victoire de grandes équipes mais qu'ils ne savent estimer les
probabilités des évènements rares. Nous optimisons donc notre modèle pour coller aux valeurs des bookmakers sur les
grandes équipes. Nous pouvons ensuite estimer les probabilités des évènements rares grâce à notre modélisation
informatique et nous pouvons alors parier sur les évènements qui sont mal côtés.

Pour optimiser les paramètres du modèle nous utilisons une méthode de descente de gradient

"""

x0 = [8.0/(8+13),2.0/7,2./9,1./6,1./101]
name = ["Chelsea","ManCity","Arsenal","ManU","Tottenham"]

Clubs=["Chelsea","ManCity","Arsenal","ManU","Tottenham","Liverpool","Southampton","Swansea","Stoke","CrystalPalace","Everton","WestHam","WBA","Leicester","Newcastle","Sunderland","AstonVilla","Bournemouth","Watford","Norwich"]
#values=score the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose values are the scores of in 2014-2015 in Championship devided by 1.8 to the power kappa
Values_Points=np.array([87,79,75,70,64,62,60,56,54,48,47,47,44,41,39,38,38,90/1.8,89/1.8,86/1.8])


def V(kappa):
    return Values_Points**kappa

def get_probs(kappa,n):
    nu=V(kappa)
    ans = np.zeros(len(name))
    for i in range(n):
        ranks = sc.who_wins(sc.championnat(nu))
        for club in name:
            nbr = sc.get_number_with_name(club)
            if(ranks[nbr]<0.5):
                ans[nbr]+=1
    return ans/n

def cout(kappa,n):
    return np.sum((get_probs(kappa,n)-x0)**2)

def gradient_step(kappa,delta,alpha,n):
    current = cout(kappa,n)
    return kappa - alpha*(cout(kappa+delta,n)-current)/delta, current

def descent(kappa0, delta, alpha, n, limit, bound = 100):
    kappa = kappa0
    kappa_memory = 0.
    compteur = 0
    i=0
    t = np.zeros(limit)
    cout_vector = np.zeros(limit)
    while(compteur<5 or i<limit):
        print("i = "+str(i))
        print("kappa: "+str(kappa))
        kappa_memory = kappa
        kappa, cout = gradient_step(kappa,delta,alpha,n)
        t[i], cout_vector[i] = i,cout
        if(abs(kappa-kappa_memory)<bound):
            compteur+=1
        i+=1

    plt.plot(t,cout_vector)
    plt.show()
    return kappa

kappa0 = 1.6
delta = 0.05
alpha = 5
n = 50000
limit = 10
bound = 0.001

descent(kappa0, delta, alpha, n, limit, bound = 100)

def graph(K,n):
    t=np.linspace(1.4,2.,K)
    loss = np.zeros(K)
    for i in range(K):
        print(i)
        loss[i] = cout(t[i],n)
    plt.plot(t,loss)
    plt.show()

#graph(100,10000)