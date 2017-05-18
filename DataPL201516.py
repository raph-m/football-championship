import numpy as np
import numpy.random as rd

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
c = len(Clubs)

if mode:
    V = Values_Points
else:
    V = Values_Ranking

""""""
def championnat(nu):
    #on calcules les probabilités de victoire définies par le modèle
    winprobs = [[(nu[j])/(nu[i] + nu[j]) for i in range(c)] for j in range(c)]
    #on simule le match aller
    aller = rd.binomial(1, winprobs, [c, c])
    aller = symetrize(aller)
    #le match retour
    retour = rd.binomial(1, winprobs, [c, c])
    retour = symetrize(retour)
    #on combine les deux

    #result = aller + retour
    result = aller

    return result

def symetrize(x):
    ans = x
    for i in range(c):
        x[i][i] = 0
    for i in range(c):
        for j in range(i):
            ans[i][j] = 1-ans[j][i]
    return ans

def who_wins(result):
    current = np.sum(result,axis=1)
    sorted = np.transpose(np.sort(current))
    sorted = np.flip(sorted,axis=0)
    order = np.zeros(c)
    for i in range(c):
        order[i] = rank(current[i],sorted)
    return order

def rank(u,sorted):
    ans = 0
    indic = True
    while(indic):
        if(sorted[ans]<=u):
            indic = False
        else:
            ans+=1
    return ans

def get_number_with_name(name):
    for i in range(c):
        if(Clubs[i]==name):
            return i

def winning_championship_proba(nu,n,name):
    sigma = 0.0
    number = get_number_with_name(name)
    for i in range(n):
        current = who_wins(championnat(nu))
        if(current[number] < 0.5):
            sigma+=1
    return sigma/n

print(who_wins(championnat(V)))
n=3000

#for name in ["Chelsea", "ManCity", "Arsenal", "ManU", "Tottenham"]:
 #  print("proba of winning for "+name+" with n = "+str(n)+": "+str(winning_championship_proba(V,n,name)))

def wins(result,name):
    current = who_wins(result)
    if(current[get_number_with_name(name)]<0.5):
        return 1
    else:
        return 0

def rare_event_basic(name,m):
    nu = V.copy()
    number = get_number_with_name(name)
    nu[number] = np.sum(V)/10
    sigma = 0.0
    for i in range(m):
        result = championnat(nu)
        if(wins(result,name)==1):
            sigma += facteur(result,V,nu)
    return sigma/m


def p(i,j,nu):
    return nu[j]/(nu[i]+nu[j])

def un_facteur(result,nu,nu_prime,i,j):
    return (1-p(i,j,nu))*((p(i,j,nu)*(1-p(i,j,nu_prime))/(p(i,j,nu_prime)*(1-p(i,j,nu))))**result[i][j])/(1-p(i,j,nu_prime))

def facteur(result,nu,nu_prime):
    ans = 1.0
    for i in range(c):
        for j in range(i):
            ans*=un_facteur(result,nu,nu_prime,i,j)
    return ans

print(rare_event_basic("Norwich",1000))