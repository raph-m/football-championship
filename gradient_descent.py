import Simulation_Championnat as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


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

def descent(kappa0, delta, alpha, n, limit, bound = 0.):
    kappa = kappa0
    kappa_memory = 0.
    compteur = 0
    i=0
    t = np.zeros(limit)
    kappa_vector =np.zeros(limit)
    cout_vector = np.zeros(limit)
    while(compteur<5 and i<limit):
        print("i = "+str(i))
        print("kappa: "+str(kappa))
        kappa_memory = kappa
        kappa, cout = gradient_step(kappa,delta,alpha,n)
        t[i], cout_vector[i] = i,cout
        kappa_vector[i] = kappa
        if(abs(kappa-kappa_memory)<bound):
            compteur+=1
        i+=1

    plt.plot(t,cout_vector)
    print("mean of the last hundred: ")
    print(np.mean(kappa_vector[limit-100:]))
    plt.show()
    return kappa

kappa0 = 1.73
delta = 0.05
alpha = 0.03
n = 1000
limit = 500
bound = 0.

descent(kappa0, delta, alpha, n, limit)

def get_variance_of_loss(kappa, n):
    return np.std([((get_probs(kappa,1)-x0)**2) for i in range(n)])
var = 0.222515817208

def graph(K,n):
    t=np.linspace(1.2,2.,K)
    loss = np.zeros(K)
    for i in range(K):
        print(i)
        loss[i] = cout(t[i],n)
    plt.plot(t,loss,'b')
    #error = var * 1.96/np.sqrt(n)
    #plt.plot(t,loss-error,'g')
    #plt.plot(t,loss+error,'g')
    plt.title("evolution of loss with "+str(n)+" simulations for each point")
    plt.xlabel("kappa")
    plt.ylabel("loss")
    plt.show()
    #np.savetxt('t',t)
    #np.savetxt('loss',loss)

#graph(100,5000)


def loss_regression(t,loss,a,b,c,d):
    return np.sum((a+b*t+c*t**2+d*t**3-loss)**2)
    #ici on pourrait rajouter éventuellement une punition pour être sorti de l'intervalle de confiance

def regression(t,loss, step = 0.001, bound = 100000 ,a=0.0,b= 0.0, c = 0.0 , d=0.0, delta = 0.000001):
    a=np.mean(loss)
    for i in range(bound):
        print("new")
        print(loss_regression(t,loss,a,b,c,d))
        for i in range(1):
            a = a - step*(loss_regression(t,loss,a+delta,b,c,d)-loss_regression(t,loss,a,b,c,d))/delta
        for i in range(1):
            b = b - step*(loss_regression(t,loss,a,b+delta,c,d)-loss_regression(t,loss,a,b,c,d))/delta
        for i in range(1):
            c = c - step*(loss_regression(t,loss,a,b,c+delta,d)-loss_regression(t,loss,a,b,c,d))/delta
        for i in range(1):
            d = d - step*(loss_regression(t,loss,a,b,c,d+delta)-loss_regression(t,loss,a,b,c,d))/delta
    return a,b,c,d

def degre3(a,b,c,d,t):
    return a+b*t+c*t**2+d*t**3

def regression2(bound,a,b,c,d,kappa0,step,delta = 0.0001):
    for i in range(bound):
        kappa0 = kappa0 - step*(degre3(a,b,c,d,kappa0+delta)-degre3(a,b,c,d,kappa0))/delta
    return kappa0




""" les lignes de code qui ont permis l'obtention des tracés de la régression linéaire
n=50000
t=np.loadtxt('t')
loss = np.loadtxt('loss')

#a,b,c,d = regression(t,loss, step = 0.5, bound = 100 ,a=-1.,b= 1., c = 1.0 , d=1.0, delta = 0.01)
a,b,c,d = regression(t,loss)

print("kappa opt")
print(regression2(100000,a,b,c,d,1.7,0.01))


loss_smoothed = a+b*t+c*t**2+d*t**3

plt.plot(t,loss)
plt.plot(t,loss_smoothed,'r')
error = var * 1.96 / np.sqrt(n)
plt.plot(t,loss-error,'g')
plt.plot(t,loss+error,'g')
plt.title("régressions linéaire avec l'intervalle de confiance en vert avec n = "+str(n))
plt.xlabel("kappa")
plt.ylabel("coût")
plt.show()
"""