import csv
from collections import Counter
from math import sqrt
from collections import defaultdict

def distance(v1,v2): #äèñòàíöèÿ ìåæäó òî÷êàìè
    return sqrt(sum((v1i-v2i)**2 for v1i,v2i in zip(v1,v2)))
def majority_vote(labels): #ôóíêöèÿ îòáîðà ïî áîëüøèíñòâó ãîëîñîâ-ìåòîê áóäåò âûãëÿäåòü òàê:
    # ìåòêè óïîðÿäî÷åíû îò áëèæíåé ê äàëüíåé
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1]) # ïûòàåìñÿ ñíîâà 
def knn_classify(k, labeled_points, new_point): #êëàññèôèêàöèÿ ïî ìåòîäó k áëèæàéøèõ ñîñåäåé 
    # sort first
    by_distance = sorted(labeled_points, key = lambda point: distance(point[0], new_point))
    # k nearest find
    k_nearest_labels = [label for _,label in by_distance[:k]]
    # vote & return
    return majority_vote(k_nearest_labels)

def listenum(lst): #âûäà÷à óíèêàëüíîé ÷èñëîâîé ìåòêè êàæäîìó óíèêàëüíîìó çíà÷åíèþ äàííûõ
    temp = defaultdict(lambda: len(temp))
    res = [temp[ele] for ele in lst]
    return res

#---------------------×òåíèå ôàéëà csv----------------------
dataset = []
with open("adult.csv", encoding = 'utf-8') as ex_file:
    reader = csv.reader(ex_file, delimiter = ",")
    for row in reader:
        dataset.append(row)
#-----------------------------------------------------------  

dataset = dataset[1:] #âûêèäûâàþ çàãîëîâêè
dataset = [ds[0:2]+ds[4:] for ds in dataset] #óäàëÿþ íåêîòîðûå ñòîëáöû
#-------------------ïðåîáðàçîâàíèå äàííûõ------------------
workclass = [ds[1] for ds in dataset]
workclass = listenum(workclass)
maritalst = [ds[3] for ds in dataset]
maritalst = listenum(maritalst)
occupation = [ds[4] for ds in dataset]
occupation = listenum(occupation)
relationship = [ds[5] for ds in dataset]
relationship = listenum(relationship)
race = [ds[6] for ds in dataset]
race = listenum(race)
gender = [ds[7] for ds in dataset]
gender = listenum(gender)
nativecountry = [ds[11] for ds in dataset]
nativecountry = listenum(nativecountry)
#----------------------------------------------------------

#-------------------Êîìïàíîâêà íîâîãî ñïèñêà ïàð èç çíà÷åíèé è èòîãîâîãî çàðàáîòêà-----------------------------------------------------------------------------------------
n = 0
for i in dataset:
    dataset[n] = [i[0]] + [workclass[n]] + [i[2]] + [maritalst[n]] + [occupation[n]] + [relationship[n]] + [race[n]] + [gender[n]] + i[8:11] + [nativecountry[n]] + [i[12]]
    n+=1
ndata = [[int(dsi) for dsi in ds[:12]] for ds in dataset]
print(ndata[0:20])
df = []
for i in range(len(dataset)):
    df.append([ndata[i], dataset[i][-1]])
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------êîë-âî ñîñåäåé äëÿ õîðîøåãî ïðåäñêàçàíèÿ--------------------
for k in range(1,7):
    n_correct = 0
    for income in df:
        values, actual_inc = income
        other_incomes = [other_income for other_income in df if other_income != income]
        predicted_inc = knn_classify(k, other_incomes, values)
        if predicted_inc == actual_inc:
            n_correct +=1
    print(k, "ñîñåäåé:",n_correct,"ïðàâèëüíûõ èç",len(df))
#--------------------------------------------------------------------------------------

'''êëàññ, êîòîðûé áóäåò ó 30-ëåòíåé òåìíîêîæåé àìåðèêàíêè, ðàáîòàþùåé â ôåäåðàëüíîì ó÷ðåæäåíèè ïî 35 ÷àñîâ â íåäåëþ, îêîí÷åâøåé ñòàðøóþ øêîëó,
íå ñîñòîÿùåé â áðàêå è íå èìåþùåé ñåìüþ, çàíèìàþùåéñÿ îñòàëüíûì ñåðâèñîì, íå èìåþùàÿ íè ïðèáàâêè, íè ïîòåðè êàïèòàëà'''
print(knn_classify(3, df, [30, 4, 9, 0, 4, 2, 0, 1, 0, 0, 35, 0]))
