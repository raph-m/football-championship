import numpy as np

kappa=1.5
b,theta=23,2.

#PL season 2015/16
Clubs=["Chelsea","ManCity","Arsenal","ManU","Tottenham","Liverpool","Southampton","Swansea","Stoke","CrystalPalace","Everton","WestHam","WBA","Leicester","Newcastle","Sunderland","AstonVilla","Bournemouth","Watford","Norwich"]
#values=score the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose values are the scores of in 2014-2015 in Championship devided by 1.8 to the power kappa
Values_Points=np.array([87,79,75,70,64,62,60,56,54,48,47,47,44,41,39,38,38,90/1.8,89/1.8,86/1.8])**kappa
#values=ranking (from 20 : winner to 4: last team maintained in PL) top the power theta in the previous season to the power kappa, except for those promoted from Championship ("Bournemouth","Watford","Norwich"), whose rank is 6=20-14 (to the power theta)
Values_Ranking=np.array(b-(1+np.array(range(17)+[14,14,14])))**theta