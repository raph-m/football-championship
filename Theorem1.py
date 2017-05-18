import Simulation_Championnat as sim
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

"""illustration of theorem 1 from R. Chetrite, R. Diel, M. Lerasle The number of potential winners in
Bradley-Terry model in random environment, Ann. Appl. Probab., 2016.
This theorem says that
"""


kappa=1.5
b,theta=23,2.

#PL season 2015/16
Clubs=["Chelsea","ManCity","Arsenal","ManU","Tottenham","Liverpool","Southampton","Swansea","Stoke","CrystalPalace","Everton","WestHam","WBA","Leicester","Newcastle","Sunderland","AstonVilla","Bournemouth","Watford","Norwich"]
#values=score the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose values are the scores of in 2014-2015 in Championship devided by 1.8 to the power kappa
Values_Points=np.array([87,79,75,70,64,62,60,56,54,48,47,47,44,41,39,38,38,90/1.8,89/1.8,86/1.8])**kappa
#Chelsea is 87
#values=ranking (from 20 : winner to 4: last team maintained in PL) top the power theta in the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose rank is 6=20-14 (to the power theta)
Values_Ranking=np.array(b-(1+np.array(range(17))))**theta

"""exponential distribution on "strengths" """

#we define the number of sims

n = 100

#we define the scale

scale = 1

#this function computes the number of wins for each team, the teams being represented by their rank in the model
#we model the "strengths" with an exponential distribution

def number_of_wins_exp(N):

    #we generate the "strengths" from the exponential distribution
    wins = np.zeros(N)
    clubs = range(1, N + 1)
    for i in range(n):
        #we simulate a championship and count the number of wins for each rank
        V = rd.exponential(scale, N)
        V = np.sort(V)
        V = np.flip(V, axis = 0)
        result = sim.who_wins(sim.championnat(V))
        j = 0
        while result[j] > 0.5:
            j += 1
        wins[j] += 1
    #plt.bar(clubs, wins, width=0.5, label = "number of wins", align = "center" )
    #plt.legend(loc = "upper left")
    #plt.title( "number of wins by rank for n simulations for exp distrib on strengths", loc = "bottom")
    #plt.show()
    return wins

#on retourne la probabilité que le gagnant soit l'équipe la plus forte d'apres le modèle

def prob_strength_wins(wins):

    return wins[0]/np.sum(wins)

#pour illustrer les résultats du théorème il faut faire tendre N vers l'infini
#on simule jusqu'a N = "bound" par saut de "increment" et on plot.

def N_to_infinity(bound, increment):

    N = bound
    x = [(i+1) * increment for i in range(int(np.floor(N/increment)))]
    y = np.zeros(len(x))
    for i in range(len(x)):
        print(i)
        y[i] = prob_strength_wins(number_of_wins_exp(x[i]))
    plt.plot(x, y, label = "probability that N°1 wins")
    plt.show()






