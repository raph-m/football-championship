import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

kappa=1.5
#kappa = 1.73581251742
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

"""fonction prenant en entrée des valeurs intrinsèques et simulant un résultat (résultat sous forme d'une matrice)
On prend ici en compte les matchs retours"""
def championnat_avec_retour(nu):
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

    result = aller + retour

    return result

"""idem sans matchs retours"""
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

    result = aller

    return result

""""prend en entrée une matrice et sort cette même matrice mais en mettant des 0 sur la diago et 'symétrise' la partie inférieure gauche"""
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

""""un premier calcul de proba: on simule n championnats et on calcule la proba que le club gagne
ici comme dans la plupart des méthodes, nu représente le vecteur de valeurs intrinsèques et n le nombre de simulations"""
def winning_championship_proba(nu,n,name):
    sigma = 0.0
    number = get_number_with_name(name)
    for i in range(n):
        current = who_wins(championnat(nu))
        if(current[number] < 0.5):
            sigma+=1
    return sigma/n

"""renvoie la variance de la proba calculée juste au dessus"""
def get_variance(nu,n,name):
    number = get_number_with_name(name)
    x=np.zeros(n)
    for i in range(n):
        current = who_wins(championnat(nu))
        if(current[number] < 0.5):
            x[i]=1
    return np.std(x)


""""la première fonction g (elle dit si tel club a gagné le championnat ou pas)"""
def wins(result,name):
    current = who_wins(result)
    if(current[get_number_with_name(name)]<0.5):
        return 1
    else:
        return 0

"""idem mais en prenant en entrée un numéro d'équipe et non pas un nom"""
def wins2(result,team_number):
    current = who_wins(result)
    if(current[team_number]<0.5):
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
    print("taux de réalisation de l'évènement après échantillonage préférentiel: " + str(100*total / m)+"%")
    return sigma/m

""""une mérhode qui permet d'avoir la variance pour la variable juste au dessus"""
def get_variance_rare_event_basic(name,m):
    nu = V.copy()
    number = get_number_with_name(name)
    nu[number] = np.sum(V)/10
    sigma = 0.0
    total = 0.0
    x=np.zeros(m)
    for i in range(m):
        result = championnat(nu)
        if(wins(result,name)==1):
             x[i] = facteur(result,V,nu)
    return np.std(x)

"""une fonction auxiliaire pour récupérer les pij"""
def p(i,j,nu):
    return nu[i]/(nu[i]+nu[j])

"""uen fonction auxiliaire pour le calcul lors de l'échantillonage préférentiel"""
def un_facteur(result,nu,nu_prime,i,j):
    return (1-p(i,j,nu))*((p(i,j,nu)*(1-p(i,j,nu_prime))/(p(i,j,nu_prime)*(1-p(i,j,nu))))**result[i][j])/(1-p(i,j,nu_prime))

"""encore une fonction auxiliaire"""
def facteur(result,nu,nu_prime):
    ans = 1.0
    c=len(nu)
    for i in range(c):
        for j in range(i):
            ans*=un_facteur(result,nu,nu_prime,i,j)
    return ans

"""encore une fonction auxiliaire"""
def rare_event_complex_aux(result):
    current = who_wins(result)
    if (current[get_number_with_name("Leicester")]>0.5):
        return False
    if ((current[get_number_with_name("ManU")]<2.5) or (current[get_number_with_name("ManCity")]<2.5)):
        return False
    if (current[get_number_with_name("Liverpool")]<6.5 or current[get_number_with_name("Chelsea")]<6.5):
        return False
    return True

"""la méthode qui fait le décalage préférentiel. On doit modifier les forces intrinsèques de plusieurs équipes pour
forcer la réalisation de l'évènement"""
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
        if(i%100 == 0):
            print(i*100.0/1000000.0)
        result = championnat(nu)
        if(rare_event_complex_aux(result)):
            total = total + 1
            sigma += facteur(result,V,nu)
    print("taux de réalisation de l'évènement après échantillonage préférentiel: "+str(100*total/m)+"%")
    return sigma/m

"""renvoie le numéro de l'équipe qui gagne le championnat"""
def who_wins2(result):
    current = np.sum(result,axis=1)
    max = 0
    best = 0
    for i in range(len(current)):
        if(current[i]>=max):
            max=current[i]
            best = i
    return best

"""permet de générer des vecteurs nu selon des distributions précises"""
def get_teams(N,law,a,b):
    if(law=="uniform"):
        return np.flip(np.sort(np.random.rand(N)),axis=0),1.0
    if(law=="beta"):
        return np.flip(np.sort(np.random.beta(a,b,N)),axis=0),b

"""une fonction qui produit un tracé illustrant le théorème 2"""
def theorem_2(N,n,law,a,b):
    teams, alpha = get_teams(N,law,a,b)
    gamma_size = 100
    gamma = np.linspace(0.0,1.0,gamma_size)
    probas = np.zeros(gamma_size)
    for i in range(n):
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

"""une fonction qui permet d'illustrer le théorème 3 (un exemple concret)"""
def theorem3(N,n,strength):
    disparate = np.ones(N+1)*0.8
    disparate[0] = strength
    disparate[1] = 1.5
    uniform = np.ones(N+1)
    uniform[0] = strength

    sigma1=0
    sigma2=0
    for i in range(n):
        result2 = championnat(disparate)
        result1 = championnat(uniform)
        if (who_wins2(result1)==0):
            sigma1+=1
        if (who_wins2(result2)==0):
            sigma2+=1
    ans1 = 100*sigma1/n
    ans2 = 100*sigma2/n
    return ans1,ans2

"""une fonction qui permet de v(u) pour la distribution uniforme"""
def get_vu():
    u=np.random.rand(10000)
    return np.mean(u/((u+1)**2))

"""une fonction auxiliaire pour calculer epsilon N"""
def get_eps(N,alpha):
    return  np.sqrt((2-alpha)*np.log(N)/(N*get_vu()))

"""une fonction qui produit une illustration du théorème 3"""
def theorem3_bis(N,n,coef):
    alpha = 1.
    team = np.random.rand(N+1)
    team[N] = 1+(1.+coef)*get_eps(N,alpha)
    print("strength: "+str(team[N]))
    sigma=0.
    for i in range(n):
        result = championnat(team)
        if (who_wins2(result) == N):
            sigma+=1
    return 100*sigma/n

"""une fonction auxiliaire qui génère une liste avec tous matchs possibles"""
def create_population(nu):
    c=len(nu)
    ans = []
    for j in range(1,c):
        for i in range(j):
            ans.append((i,j))
    return ans

"""une fonction qui prend un résultat puis resimule une fraction ro des matchs et renvoie le nouveau résultat"""
def modify(result,ro,nu):
    c=len(nu)
    number_of_matches = int((c**2 - c)/2)
    number_to_modify = int(ro*number_of_matches)
    matches = create_population(nu)
    np.random.shuffle(matches)
    matches_to_modify = matches[:number_to_modify]
    for match in matches_to_modify:
        i,j= match
        issue = np.random.binomial(1,nu[i] / (nu[i] + nu[j]),1)
        result[i][j] = issue
        result[j][i] = 1-issue
    return result

"""renvoie le score d'une équipe"""
def score(result,team_number):
    return np.sum(result[team_number])

""""renvoie le terme suivant dans la suite pour le splitting"""
def next_result(result,ro,bound,team_number,nu):
    ans = modify(result,ro,nu)
    if (score(ans,team_number)>= bound):
        return ans,0
    else:
        return result,1

""""probabilité que l'équipe dépasse ce score"""
def proba_score(bound_inf,team_number,nu,n):
    total = 0.
    for i in range(n):
        result = championnat(nu)
        if(score(result,team_number)>=bound_inf):
            total+=1
            last_no_rejected = result.copy()
    if (total < 0.5):
        print("attention aucun résultat convenable n'a été trouvé pour la première borne")
    return total/n, last_no_rejected

"""renvoie la probabilité de dépasser la borne sup conditionnée à la borne inf """
def conditionnal_proba(result,nu,ro,team_number,n,inf_bound,sup_bound=0,last_bound = False):
    taux_de_rejection = 0.
    total = 0.
    last_no_rejected = result.copy()

    for i in range(n):
        result,rej=next_result(result,ro,inf_bound,team_number,nu)
        taux_de_rejection += rej
        if (last_bound):
            total += wins2(result,team_number)
        else:
            if(score(result, team_number) >= sup_bound):
                total += 1
                last_no_rejected = result.copy()

    print("taux de rejection pour les bornes ("+str(inf_bound)+", "+str(sup_bound)+"): "+str(taux_de_rejection*100/n)+"%")
    if(total < 0.5):
        print("attention aucun résultat convenable n'a été trouvé pour les bornes ("+str(inf_bound)+", "+str(sup_bound)+")")

    print("score: "+str(score(last_no_rejected,team_number)))
    return total/n,last_no_rejected

"""idem mais avec le rang de l'équipe"""
def next_result2(result,ro,bound,team_number,nu):
    ans = modify(result,ro,nu)
    if (who_wins(result)[team_number]<= bound):
        return ans,0
    else:
        return result,1

"""idem mais avec les rangs"""
def proba_score2(bound_inf,team_number,nu,n):
    total = 0.
    for i in range(n):
        result = championnat(nu)
        if(who_wins(result)[team_number]<= bound_inf):
            total+=1
            last_no_rejected = result.copy()
    if (total < 0.5):
        print("attention aucun résultat convenable n'a été trouvé pour la première borne")
    return total/n, last_no_rejected

""""idem mais avec les rangs"""
def conditionnal_proba2(result,nu,ro,team_number,n,inf_bound,sup_bound=0,last_bound = False):
    taux_de_rejection = 0.
    total = 0.
    last_no_rejected = result.copy()

    for i in range(n):
        result,rej=next_result2(result,ro,inf_bound,team_number,nu)
        taux_de_rejection += rej
        if (last_bound):
            total += wins2(result,team_number)
        else:
            if(who_wins(result)[team_number]<= sup_bound):
                total += 1
                last_no_rejected = result.copy()

    print("taux de rejection pour les bornes ("+str(inf_bound)+", "+str(sup_bound)+"): "+str(taux_de_rejection*100/n)+"%")
    if(total < 0.5):
        print("attention aucun résultat convenable n'a été trouvé pour les bornes ("+str(inf_bound)+", "+str(sup_bound)+")")

    print("score: "+str(score(last_no_rejected,team_number)))
    return total/n,last_no_rejected





