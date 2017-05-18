import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

kappa=1.5
b,theta=23,2.

#PL season 2015/16
Clubs=["Chelsea","ManCity","Arsenal","ManU","Tottenham","Liverpool","Southampton","Swansea","Stoke","CrystalPalace","Everton","WestHam","WBA","Leicester","Newcastle","Sunderland","AstonVilla","Bournemouth","Watford","Norwich"]
#values=score the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose values are the scores of in 2014-2015 in Championship devided by 1.8 to the power kappa
Values_Points=np.array([87,79,75,70,64,62,60,56,54,48,47,47,44,41,39,38,38,90/1.8,89/1.8,86/1.8])**kappa
#Chelsea is 87
#values=ranking (from 20 : winner to 4: last team maintained in PL) top the power theta in the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose rank is 6=20-14 (to the power theta)
Values_Ranking=np.array(b-(1+np.array(range(17))))**theta


mode = 1


if mode:
    V = Values_Points
else:
    V = Values_Ranking

"""fonction prenant en entrée des valeurs intrinsèques et simulant un résultat (résultat sous forme d'une matrice)"""
def championnat(nu):
    c=len(nu)
    #on calcules les probabilités de victoire définies par le modèle
    winprobs = [[(nu[j])/(nu[i] + nu[j]) for i in range(c)] for j in range(c)]
    #on simule le match aller
    aller = rd.binomial(1, winprobs, [c, c])
    aller = symetrize(aller,c)
    #le match retour
    retour = rd.binomial(1, winprobs, [c, c])
    retour = symetrize(retour,c)
    #on combine les deux

    #result = aller + retour
    result = aller

    return result

""""prend en entrée une matrice et sort cette même matrice mais en mettant des 0 sur la diago et symétrise la partie inférieure gauche"""
def symetrize(x,c):
    ans = x
    for i in range(c):
        x[i][i] = 0
    for i in range(c):
        for j in range(i):
            ans[i][j] = 1-ans[j][i]
    return ans

""""fonction prenant en entrée un résultat et donnant le classement sous la forme d'un vecteur avec les positions de chacun des clubs"""
def who_wins(result):
    c=len(result[0])
    current = np.sum(result,axis=1)
    sorted = np.sort(current)
    sorted = np.flip(sorted,axis=0)
    order = np.zeros(c)
    for i in range(c):
        order[i] = rank(current[i],sorted)
    return order

"""une fonction auxiliaire qui permet de retrouver le score de qq dans la liste triée des scores et ainsi de connaitre son classement"""
def rank(u,sorted):
    ans = 0
    indic = True
    while(indic):
        if(sorted[ans]<=u):
            indic = False
        else:
            ans+=1
    return ans

"""une fonction pratique qui prend en entrée le nom d'un Club et donne son numéro dans la liste"""
def get_number_with_name(name):
    c=len(Clubs)
    for i in range(c):
        if(Clubs[i]==name):
            return i

""""un premier calcul de proba: on simule n championnats et on calcule la proba que le club gagne"""
def winning_championship_proba(nu,n,name):
    sigma = 0.0
    number = get_number_with_name(name)
    for i in range(n):
        current = who_wins(championnat(nu))
        if(current[number] < 0.5):
            sigma+=1
    return sigma/n

"""
for n in [3000]:
    for name in ["Chelsea", "ManCity", "Arsenal", "ManU", "Tottenham"]:
        print("proba of winning for "+name+" with n = "+str(n)+": "+str(winning_championship_proba(V,n,name)))

for x in [8.0/(8+13),2.0/7,2./9,1./6,1./101]:
    print(x)
"""

""""la première fonction g (elle dit si tel club a gagné le championnat ou pas)"""
def wins(result,name):
    current = who_wins(result)
    if(current[get_number_with_name(name)]<0.5):
        return 1
    else:
        return 0

""""un premier calcul permettant de simuler m évènements avec décalage préférentiel """
def rare_event_basic(name,m):
    nu = V.copy()
    number = get_number_with_name(name)
    nu[number] = np.sum(V)/10
    sigma = 0.0
    total = 0.0
    for i in range(m):
        result = championnat(nu)
        if(wins(result,name)==1):
            total = total + 1
            sigma += facteur(result,V,nu)
    return sigma/m

def p(i,j,nu):
    return nu[i]/(nu[i]+nu[j])

def un_facteur(result,nu,nu_prime,i,j):
    return (1-p(i,j,nu))*((p(i,j,nu)*(1-p(i,j,nu_prime))/(p(i,j,nu_prime)*(1-p(i,j,nu))))**result[i][j])/(1-p(i,j,nu_prime))

def facteur(result,nu,nu_prime):
    ans = 1.0
    c=len(nu)
    for i in range(c):
        for j in range(i):
            ans*=un_facteur(result,nu,nu_prime,i,j)
    return ans

def rare_event_complex_aux(result):
    current = who_wins(result)
    if (current[get_number_with_name("Leicester")]>0.5):
        return False
    if ((current[get_number_with_name("ManU")]<2.5) or (current[get_number_with_name("ManCity")]<2.5)):
        return False
    if (current[get_number_with_name("Liverpool")]<6.5 or current[get_number_with_name("Chelsea")]<6.5):
        return False
    return True


def rare_event_complex(m):
    nu = V.copy()
    number = get_number_with_name("Leicester")

    nu[number] = np.sum(V)/5
    nu[get_number_with_name("Chelsea")]=V[get_number_with_name("Chelsea")]/3
    nu[get_number_with_name("ManCity")]=V[get_number_with_name("ManCity")]/3
    nu[get_number_with_name("Liverpool")]=V[get_number_with_name("Liverpool")]/3
    nu[get_number_with_name("ManU")]=V[get_number_with_name("ManU")]/2

    sigma = 0.0
    total = 0.0
    for i in range(m):
        result = championnat(nu)
        if(rare_event_complex_aux(result)):
            total = total + 1
            sigma += facteur(result,V,nu)
    print("taux de réalisation: "+str(total/m))
    return sigma/m

"""
print(winning_championship_proba(V,100,"Leicester"))
print(rare_event_basic("Leicester",100))
print(rare_event_basic("Norwich",500))
"""
"""
memo = np.zeros(10)
for i in range(10)
    memo[i] = rare_event_complex(5000)
print(memo)
"""

"""illustration of theorem 1 from R. Chetrite, R. Diel, M. Lerasle The number of potential winners in
Bradley-Terry model in random environment, Ann. Appl. Probab., 2016.
This theorem says that
"""

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

""""1: loi uniforme
on va simplement faire un grand nombre de simulations et à chaque fois regarder"""

def who_wins2(result):
    current = np.sum(result,axis=1)
    max = 0
    best = 0
    for i in range(len(current)):
        if(current[i]>=max):
            max=current[i]
            best = i
    return best

def get_teams(N,law,a,b):
    if(law=="uniform"):
        return np.flip(np.sort(np.random.rand(N)),axis=0),1.0
    if(law=="beta"):
        return np.flip(np.sort(np.random.beta(a,b,N)),axis=0),b


def theorem_2(N,n,law,a,b):
    teams, alpha = get_teams(N,law,a,b)
    gamma_size = 100
    gamma = np.linspace(0.0,1.0,gamma_size)
    probas = np.zeros(gamma_size)
    for i in range(n):
        print(i)
        result = championnat(teams)
        for j in range(len(gamma)):
            if(who_wins2(result)<N**gamma[j]):
                probas[j]+=1
    plt.plot(gamma,probas/n)
    plt.axvline(1-alpha/2)
    if(law=="uniform"):
        plt.title("championship with "+str(N)+" players with a uniform distribution of strength, the probabilities were computed with "+str(n)+" different championships")

    if(law=="beta"):
        plt.title("championship with "+str(N)+" players with a beta distribution of strength, parameters are a = "+str(a)+", b = "+str(b)+" the probabilities were computed with "+str(n)+" different championships")


    plt.xlabel("gamma")
    plt.ylabel("probability of having one of the N**gamma best players winning the championship")
    plt.show()

plt.close('all')
#theorem_2(5000, 100, "uniform", 1., 1.)

def theorem3(N,n,strength):
    team1 = np.ones(N+1)
    team2 = np.ones(N+1)
    for i in range(N+1):
        if (i%2 == 0):
            team2[i]=0.5
    team1[N] = strength
    team2[N] = strength
    sigma1=0
    sigma2=0
    for i in range(n):
        print(i)
        result1 = championnat(team1)
        result2 = championnat(team2)
        if (who_wins2(result1)==N):
            sigma1+=1
        if (who_wins2(result2)==N):
            sigma2+=1
    ans1 = 100*sigma1/n
    ans2 = 100*sigma2/n
    print("pourcentage teams avec disparités: "+str(ans2))
    print("pourcentage teams uniformes: "+str(ans1))
    return ans1,ans2


#theorem3(200,100,1.5)

def illustrate_theorem3(strength):
    #Ns=[10]
    Ns = [20,50,100,200]
    disparate = np.zeros(len(Ns))
    uniform = np.zeros(len(Ns))
    i=0
    n=100
    for N in Ns:
        print(N)
        uniform[i],disparate[i] = theorem3(N,500,1.5)
        i+=1
    plt.title("illustration of theorem 3 with Strength(N+1) = "+str(strength)+" and with "+str(n)+" simulations")
    plt.xlabel("number of players")
    plt.ylabel("probability of winning for the N+1 player (red = disparate teams, blue = uniform teams")
    plt.plot(Ns,uniform)
    plt.plot(Ns,disparate,'r')
    plt.show()

#theorem3(300,100,1.2)

def get_vu():
    u=np.random.rand(10000)
    return np.mean(u/((u+1)**2))

def get_eps(N,alpha):
    return  np.sqrt((2-alpha)*np.log(N)/(N*get_vu()))

def theorem3_bis(N,n,coef):
    alpha = 1.
    vu = get_vu()
    team = np.random.rand(N+1)
    team[N] = 1+(1.+coef)*get_eps(N,alpha)
    sigma=0.
    for i in range(n):
        print(i)
        result = championnat(team)
        if (who_wins2(result) == N):
            sigma+=1
    return 100*sigma/n

print(theorem3_bis(900,100,0.5))
print(theorem3_bis(900,100,-0.5))












