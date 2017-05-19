import Simulation_Championnat as sim
import numpy as np
import numpy.random as rd

"""illustration of theorem 1 from R. Chetrite, R. Diel, M. Lerasle The number of potential winners in
Bradley-Terry model in random environment, Ann. Appl. Probab., 2016."""

kappa=1.5
b,theta=23,2.

#PL season 2015/16
Clubs=["Chelsea","ManCity","Arsenal","ManU","Tottenham","Liverpool","Southampton","Swansea","Stoke","CrystalPalace","Everton","WestHam","WBA","Leicester","Newcastle","Sunderland","AstonVilla","Bournemouth","Watford","Norwich"]
#values=score the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose values are the scores of in 2014-2015 in Championship devided by 1.8 to the power kappa
Values_Points=np.array([87,79,75,70,64,62,60,56,54,48,47,47,44,41,39,38,38,90/1.8,89/1.8,86/1.8])**kappa
#Chelsea is 87
#values=ranking (from 20 : winner to 4: last team maintained in PL) top the power theta in the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose rank is 6=20-14 (to the power theta)
Values_Ranking=np.array(b-(1+np.array(range(17))))**theta

c = len(Clubs)

#exponential distribution on "strengths"

#we define the number of sims

n=1000

#we define the scale

scale = 1


def theorem1_exp():
#we generate the "strengths" from the exponential distribution
    for i in range(n):
        V = rd.exponential(scale, c)
        V=np.sort(V)
        result = sim.who_wins(sim.championnat(V))


