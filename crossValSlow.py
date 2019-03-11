'''
Remember that it is important to add:
different starting conditions (liek are present in the clusteringSoft2.0 or clustering graph 2.0)
try to add more options allowing for soft clustering in the cross validation
'''
#minor difference here (between this and regulare cross val) is that this does not have the true clusters
import numpy as np
import math
import matplotlib.pyplot as plt
from sequenceObj import Sequence
from sequenceObj import ArrayRank
from sklearn.metrics.cluster import adjusted_rand_score
import random
import time

#go back to old probability func.?
#change trialFreq to 4, counter 1 default
trueK = 10
start_time = time.time()
# K = number of clusters
N = 50
# N = length of each sequence
S = 500
# S = number of data points
printStor = 5
#number of times to run the whole thing, and store in a file!
f1= open("text.txt", "w+")
#opens the file to write in
tenPerc = (int)( S/10)
ninetyPerc=S-tenPerc
emptyClus = []
clusCount = []

#to hold all the data from the file
masterDataSeqObj =[]

#number of times to run each cross validation (iot determine the best option)
bestOpRuns = 3

#convert to 5 fold, instead of 10 fold
#stop probability

kArr = [1,2,3,4,5,6,7,8]
#kArr = array of different clusters to try

#where to start from for the last part (S+1)

# A = 0
# C = 1
# T = 2
# G = 3

# data = [[A,A],[C,C],[T,T],[G,G],[A,A],[C,C],[T,T]]
# print(data)
# data_full = [[G,A]]
# size = 5
# for i in range(size):
# data_full = np.concatenate((data_full, data))

# print(data_full)
#this code segments reads ATCG values from a flie, and converts them to arrays of 1,2,3,4
#yep = open("/Users/mallika/PycharmProjects/DirichletBio/venv/lib/shuffledExtendedseq.txt", "r")
#writtenText is unshuffled
yep = open("/Users/mallika/PycharmProjects/DirichletBio/venv/lib/shuffledWrittenFile.txt", "r")
for line in yep:
    values = line.split()
    seq = []
    #line = yep.readline()
    for j in range(N):
        if values[0][j] == "A":
            seq.append(0)
        elif values[0][j] == "C":
            seq.append(1)
        elif values[0][j] == "T":
            seq.append(2)
        else:
            seq.append(3)
    #s = Sequence.make_sequence(int(values[1]), seq)
    s = Sequence.make_sequence(-1, seq)
    masterDataSeqObj.append(s)
yep.close()

#checking the initial logProduct
for i in range(masterDataSeqObj.__len__()):
    masterDataSeqObj[i].currentCluster=masterDataSeqObj[i].trueCluster

counter = [[]]
counter = [[1 for i in range(N)] for j in range(4*trueK)]

def counterCalc (seqOBJArr, counter):
    #iterating through the sequence array and entering the number of ACTG for each cluster. For example, in a scenario with four clusters, the table would have 4 times 4 = 16 rows and 2 columns (N)
    #resets all the rows and columns of the counter to one
    for row in range(4*trueK):
        for column in range (N):
            counter[row][column]= 1
    #recalculates the counts of each ACTG of each cluster
    for i in range (seqOBJArr.__len__()):
        k = seqOBJArr[i].currentCluster
        for j in range (N):
            m = seqOBJArr[i].value[j]
            counter [4*k + m][j] = counter [4*k + m][j] + 1

probability = [[]]
probability =  [[0 for i in range(N)] for j in range(4*trueK)]
def probabilityCalc (probability, counter):
    #goes 1 column, 1 cluster at a time and uses the counter array and a sum function calculate the probability of each ACTG
        for k in range (trueK):
            for n in range (N):
                summ = 0
                for b in range (4):
                    summ = summ + counter [4*k + b][n]
                for b in range (4):
                    probability[4*k+b][n] = counter [4*k + b][n]/summ

counterCalc(masterDataSeqObj, counter)
probabilityCalc(probability,counter)


trialTrueFreq = [4]*trueK
#the following loop calculates the trial frequencies of each cluser
for a in range (masterDataSeqObj.__len__()):
    trialTrueFreq[masterDataSeqObj[a].currentCluster]+= 1
#print("trialNum", trialFreq)

trialTrueFreqsum = 0
for a in range(trialTrueFreq.__len__()):
    trialTrueFreqsum += trialTrueFreq[a]
#find sum of trial frequencies

trialTrueFreqProb=[0]*trueK
#make trial Freq actuall a frequency by dividing it by sum
for a in range(trialTrueFreq.__len__()):
    trialTrueFreqProb[a]= trialTrueFreq[a]/trialTrueFreqsum
#(trialTrueFreqProb)

TrueTrialProb=[[0 for i in range(trueK)]for j in range(S)]
for s in range (S):
    #print("s", s)
    #print("smallLen", smallSeqOBJ.__len__())
    for k in range (trueK):
        probcurrent=0
        for n in range (N):
            seq= masterDataSeqObj[s].value[n]
            p = probability [4*k+seq][n]
            probcurrent = probcurrent + math.log10(p)
        TrueTrialProb[s][k]= probcurrent
        #print("probcurremnt",probcurrent,s)

TruetrainingWP=[]
for l in range (TrueTrialProb.__len__()):
    wProb = 0
    for k in range (trueK):
    #print(10**(TrainingTrialProb [l][k]))
        wProb += (10**(TrueTrialProb [l][k])) * trialTrueFreqProb[k]               #
    TruetrainingWP.append(math.log10(wProb))

logProd=0
for i in range(TruetrainingWP.__len__()):
    logProd+= TruetrainingWP[i]
#print("Bestlogprod", logProd)



trueClustering = [0]*trueK
for i in range(masterDataSeqObj.__len__()):
    trueClustering[masterDataSeqObj[i].trueCluster]+=1
#("trueClustering", trueClustering)
#print(masterDataSeqObj.__len__())

for pStor in range(printStor):
    weightProbMaster = []
    for arr in range (kArr.__len__()):
        K = kArr [arr]
        clustWPM=[]
        #print("K start", K)
        #weightedprobtrial will be a 10 rows by tenperc columns arr. each row contains 1 cross validation,
        # where each element is the sum of the K-clusterprobs
        weightedProbTrial = []
        for repeat in range (10):
            #print("repeat", K, repeat)
            #This stores the likelihoods for each of the training data, where the highest one
            #  will determine which corresponding cross validation to pick
            sumTestArrBest = []

            #storing the cluster assignments and sumArr for the three clusterings for each cross validation,
            # the best one eventually goes into weightedProbtrial
            trialProbStorage = []
             #the next loop is a loop that finds the best clustering option for the data, by
            #finding the one that best fits the training data
            for bestOp in range(bestOpRuns):
                print("bestOp", K, bestOp)
                #temtrial will store the weighted probs
                tempTrial = []
                #yep = open("/Users/mallika/PycharmProjects/DirichletBio/venv/lib/mALphashuffle.txt", "r")
                #yep = open("/Users/mallika/PycharmProjects/DirichletBio/venv/lib/shuffledwrittenFile.txt", "r")
                finallist = []
                seqOBJArr = []
                linNum = 0
                for line in range(masterDataSeqObj.__len__()):
                    #checks if the linenumber is greater than T, and only then looks at it
                    if ((linNum < repeat*tenPerc) or (linNum > (repeat+1)*tenPerc-1)):
                        seq=masterDataSeqObj[line]
                        seqOBJArr.append(seq)
                        #setting truecluster to -1, cause we don't know it here
                        #putting in true cluster for now, take it out for TB
                    linNum+= 1

                cluster_id = []
                # the following code creates the default cluster id array, with each sequence being assigned a random number from 1-n clusters
                #temp. not using a random number
                for i in range (seqOBJArr.__len__()):
                    #c = (int) (i/(seqOBJArr.__len__()/K))
                    #print("i", i, "seq", seqOBJArr.__len__(), K, "c", c)
                    c = random.randint(0,(K-1))
                    seqOBJArr[i].currentCluster = c
                #matrix = [[]]
                #matrix = [[0 for i in range(a)] for i in range(b)]
                #creates a 2D array with b rows and a columns, set to O


                counter = [[]]
                counter = [[1 for i in range(N)] for i in range(4*K)]

                probability = [[]]
                probability =  [[0 for i in range(N)] for i in range(4*K)]

                def counterCalc (seqOBJArrArg, counterArg, initValue):
                    #initValue is either 4 or 0.25 depends
                #iterating through the sequence array and entering the number of ACTG for each cluster. For example, in a scenario with four clusters, the table would have 4 times 4 = 16 rows and 2 columns (N)
                    #resets all the rows and columns of the counter to one
                    for row in range(4*K):
                        for column in range (N):
                            counterArg[row][column]= initValue
                    #recalculates the counts of each ACTG of each cluster
                    for i in range (seqOBJArrArg.__len__()):
                        k = seqOBJArrArg[i].currentCluster
                        for j in range (N):
                            m = seqOBJArrArg[i].value[j]
                            counterArg[4*k + m][j] = counterArg[4*k + m][j] + 1

                def probabilityCalc (probabilityArg, counterArg):
                #goes 1 column, 1 cluster at a time and uses the counter array and a sum function calculate the probability of each ACTG
                    for k in range (K):
                        for n in range (N):
                            summ = 0
                            for b in range (4):
                                summ = summ + counterArg[4*k + b][n]
                            for b in range (4):
                                probabilityArg[4*k+b][n] = counterArg[4*k + b][n]/summ

                def ClusterCalc(seqOBJArrArg, probabilityArg, counterArg):
                    changeCount = 10
                    while (changeCount != 0):
                        changeCount = 0
                        #counts the number of times a sequence's cluster is reassigned
                        counterCalc(seqOBJArrArg, counterArg, 1)
                        probabilityCalc (probabilityArg, counterArg)
                        # print("cluster: ",cluster_id)
                        # print("probability: ", probability)
                        for s in range (seqOBJArrArg.__len__()):
                            prob = -1000000000000000
                            optimclust = -1
                            for k in range (K):
                                probcurrent= 0
                                for n in range (N):
                                    seq=seqOBJArrArg[s].value[n]
                                    p = probabilityArg[4*k+seq][n]
                                    probcurrent = probcurrent + math.log10(p)
                                if (probcurrent > prob):
                                    #print("probcurrent: ",probcurrent," prob: ", prob)
                                    prob = probcurrent
                                    optimclust = k

                            old_cluster_id1 = seqOBJArrArg[s].currentCluster
                            #print("old :", old_cluster_id1)
                            seqOBJArrArg[s].currentCluster= optimclust
                            #print("optim :", optimclust)
                            #old_cluster_id = cluster_id[s]
                            #print("old2 :", old_cluster_id1)
                            #cluster_id[s]= optimclust
                            #print("optim2 :", optimclust)
                            #print(cluster_id[s])
                            #print(optimclust, old_cluster_id1)
                            if (optimclust !=  old_cluster_id1):
                                changeCount = changeCount + 1
                                #print("change:", changeCount)
                        #print("changeCount: ", changeCount)
                        #checking if there were no swaps made:

                ClusterCalc(seqOBJArr, probability, counter)
                trueClusterarr=[-1]*seqOBJArr.__len__()
                for i in range(seqOBJArr.__len__()):
                    trueClusterarr[i]=seqOBJArr[i].trueCluster
                currentClusterarr=[-1]*seqOBJArr.__len__()
                for i in range(seqOBJArr.__len__()):
                    currentClusterarr[i]=seqOBJArr[i].currentCluster
                #print("ARI", "K", K, "repeat", repeat, (adjusted_rand_score(currentClusterarr, trueClusterarr)))
                #print(probability)
                #checking if there are any empty clusters, and tallying total num in each clus
                clusCountSmall = []
                emptyClusSmall = []
                for clus in range(K):
                    Count = 0
                    for seq in range(seqOBJArr.__len__()):
                        if (seqOBJArr[seq].currentCluster == clus ):
                            Count += 1
                    clusCountSmall.append(Count)
                    if (Count == 0):
                        emptyClusSmall.append(K)
                emptyClus.append(emptyClusSmall)
                clusCount.append(clusCountSmall)
                #empty array of length K (cluster number)
                trialTrainFreq = [4]*K
                #the following loop calculates the trial frequencies of each cluser
                for a in range (seqOBJArr.__len__()):
                    trialTrainFreq[seqOBJArr[a].currentCluster]+= 1
                #print("trialNum", trialFreq)

                trialTrainFreqsum = 0
                for a in range(trialTrainFreq.__len__()):
                    trialTrainFreqsum += trialTrainFreq[a]
                #find sum of trial frequencies

                trialTrainFreqProb=[0]*K
                #make trial Freq actuall a frequency by dividing it by sum
                for a in range(trialTrainFreq.__len__()):
                    trialTrainFreqProb[a]= trialTrainFreq[a]/trialTrainFreqsum
                #print("trialFreq", trialFreq)
                #calculating the probability of the 90% existing within the given model
                #same code as the ten perc
                #S-tenPerc is basically the 90%
                counterCalc(seqOBJArr,counter,1)
                probabilityCalc(probability,counter)
                TrainingTrialProb=[[0 for i in range(K)]for j in range(ninetyPerc)]
                for s in range (ninetyPerc):
                    #print("s", s)
                    #print("smallLen", smallSeqOBJ.__len__())
                    for k in range (K):
                        probcurrent=0
                        for n in range (N):
                            seq= seqOBJArr[s].value[n]
                            p = probability [4*k+seq][n]
                            probcurrent = probcurrent + math.log10(p)
                        TrainingTrialProb[s][k]= probcurrent

                trainingWP = [0]*(ninetyPerc)

                for s in range (TrainingTrialProb.__len__()):
                    wProb = 0
                    for k in range (K):
                        #print(10**(TrainingTrialProb [l][k]))
                        wProb += (10**(TrainingTrialProb [s][k])) * trialTrainFreqProb[k]
                        #print("prob= ", (TrainingTrialProb [l][k]), "*", "trialFreq",trialTrainFreqProb[k], "final product", wProb, "wPsum", WPsum)
                        #10 raised to to convert trial freq to probability (it is currently logs)
                    trainingWP[s]= math.log10(wProb)
                #adding the prob in log space (ie multiplying themm)
                logProd=0
                for i in range(trainingWP.__len__()):
                    logProd+= trainingWP[i]
                #print("logprod", logProd)
                sumTestArrBest.append(logProd)
                #print("logSum", logProd, "K", K, "bestop", bestOp)


                smallSeqOBJ= []
                linNum = 0
                for line in range(masterDataSeqObj.__len__()):
                    #extracting the 10%
                    if (linNum >= tenPerc*repeat and linNum < tenPerc*(repeat+1)):
                        seq = masterDataSeqObj[linNum]
                        smallSeqOBJ.append(seq)
                    linNum += 1
                #print(smallSeqOBJ.__len__(),repeat )
                #calculate the probability of belonging to a cluster

                #create an array of K columns and length S-T
                #matrix = [[0 for i in range(a)] for i in range(b)]
                #creates a 2D array with b rows and a columns, set to O
                trialProb=[[0 for i in range(K)]for i in range(tenPerc)]
                for t in range (tenPerc):
                    #print("s", s)
                    #print("smallLen", smallSeqOBJ.__len__())
                    prob = -1000000000000000
                    optimclust = -1
                    for k in range (K):
                        probcurrent= 0
                        for n in range (N):
                            seq = smallSeqOBJ[t].value[n]
                            p = probability [4*k+seq][n]
                            probcurrent = probcurrent + math.log10(p)
                        trialProb[t][k]= probcurrent

                weightedProbability = [0]*(tenPerc)
                for t in range (trialProb.__len__()):
                    wProb = 0
                    for k in range (K):
                        wProb += (10**(trialProb [t][k])) * trialTrainFreqProb[k]
                        #if ((K==13 or K==10)and (repeat==1)):
                           #print("K", K, "k", k, "repeat", repeat, "seq", l, "trialProb",(10**(trialProb [l][k])) * trialTrainFreqProb[k], "*" , "trialFreq", trialTrainFreq[k] )
                    #if ((K==13)or(K==10)):
                        #print("K", K, "k", k, "seq", l, "wProbsum", wProb, "repeat", repeat )
                        #print("trialProb", trialProb [l][k], "*", "trialfreq", trialFreq[k], "product", (10**(trialProb [l][k])) * trialFreq[k],  "Cluster# ", k)
                        #print("prob= ", 10**(trialProb [l][k]))
                        #10 raised to to convert trial freq to probability (it is currently logs)
                    weightedProbability[t]= wProb
                    #print("wProb", wProb)
                trialProbStorage.append(weightedProbability)
                #print("30sum", weightedProbability)
                #print("wP", weightedProbability)

            #seeing which model was ideal of the three
            #print("sumTestArr", sumTestArrBest)
            greatest = 0
            greatestPos = -1
            for i in range(sumTestArrBest.__len__()):
                if (sumTestArrBest[i]> greatest):
                    greatest= sumTestArrBest[i]
                    greatestPos=i
            #print("bestTRain", greatest)


            #now using the greatest pos, to find the right weightedprob trial version, and then add to weightedprobtrial
            #print("greeatestPpos", greatestPos)
            #if ((K==3 or K==4)):
                #print("trialProbLen", trialProbStorage[greatestPos], "repeat", repeat)
            weightedProbTrial.append(trialProbStorage[greatestPos])
            #print("best", trialProbStorage[greatestPos])
            #print("idealfromtrialstore", trialProbStorage[greatestPos].__len__())
        #MULTIPLYING up all the 10 different sums (from each cluster validation), by converting into logspace and adding
        # of each sequence in the testing (total is 170), and then averaging them,
        weightProbAvg = []
        #print("weightedProbtria;", weightedProbTrial)
        #print("wpt", weightedProbTrial)
        for row in range(weightedProbTrial.__len__()):
            Lsum= 0
            for column in range(weightedProbTrial[0].__len__()):
                Lsum = Lsum + math.log10(weightedProbTrial[row][column])
                #print("10sum", weightedProbTrial[a][j], "clusteringSys", K)
                #print("weight", weightedProbTrial[a][j])
            #if ((K==3 or K==6)):
                #print("avg", avg)
                #print("wpLen", weightedProbTrial[a].__len__())
            weightProbAvg.append(Lsum)

            #print("K", K, "avg", avg, "a", a)

        weightProbMaster.append(weightProbAvg)
        #print("wPA", weightProbAvg, "K", K)
        ##### commented by Mama wPMadder.append(clustWPM)




    #print("wPM", weightProbMaster)




    sumArr = [0]*(weightProbMaster.__len__())

    for r in range(weightProbMaster.__len__()):
        sum = 0
        for c in range(weightProbMaster[0].__len__()):
            sum = sum + weightProbMaster[r][c]
            #print(sum, "r", r, kArr[r])
        sumArr[r]= sum

    #way 2 of evaluating (max counts)
    #maxCounts arr stores the number of times a wPa max is from that cluster sys.
    #print("wpm", weightProbMaster)
    maxCounts=[0]*kArr.__len__()
    for mcCol in range(weightProbMaster[0].__len__()):
        max = -1000000000000000
        maxPos=-1
        for mcRow in range(weightProbMaster.__len__()):
            current = weightProbMaster[mcRow][mcCol]
            if (current > max):
                max = current
                maxPos= mcRow
                #print("max", max)
                #print("pos", mcRow)
        maxCounts[maxPos]+=1
    #print("wpm", weightProbMaster)


    #makes objects in order to help rank k by way2
    #kArr is all the arrays being tested on, maxcounts is the number of times each of them got a max count
    arrayObjArr = []
    for ctr in range(kArr.__len__()):
        s = ArrayRank.makearrObj (kArr[ctr],maxCounts[ctr] )
        arrayObjArr.append(s)


    #sort the max counts in descending order
    arrayObjArrCopy = arrayObjArr
    for j in range(arrayObjArrCopy.__len__()):
        for i in range(arrayObjArrCopy.__len__()-j-1):
            if (arrayObjArr[i].bestNum<arrayObjArr[i+1].bestNum):
                temp = arrayObjArr[i]
                arrayObjArr[i]= arrayObjArr[i+1]
                arrayObjArr[i+1]= temp

    sortedWay2Arr = []
    for a in range(arrayObjArrCopy.__len__()):
        sortedWay2Arr.append(arrayObjArrCopy[a].k)








    #print("sumArr", sumArr)

    '''
    #following code only works for when you are comparing 2 arrays
    betterArr = [0]*(weightProbMaster.__len__())
    #checks for each sequence which clustering was better
    for c in range(S-T):
        weights = []
        for i in range(weightProbMaster.__len__()):
            weights.append(weightProbMaster[i][c])
    
        if (weights[0]>weights[1]):
            betterArr[0]+=1
        if (weights[1]>weights[0]):
            betterArr[1]+=1
    
    
    print(betterArr)
    '''

    #sorts the array copy- using bubble sort ayyy
    #in descending order--best to worst
    sumArrCopy = sumArr.copy()
    for j in range(sumArrCopy.__len__()):
        for i in range(sumArrCopy.__len__()-j-1):
            if (sumArrCopy[i]<sumArrCopy[i+1]):
                temp = sumArrCopy[i]
                sumArrCopy[i]= sumArrCopy[i+1]
                sumArrCopy[i+1]= temp

    #
    print("copy", sumArrCopy)

    arrayRanking = [0]*(sumArrCopy.__len__())
    #set to one from the start to since we are testing cluster numbers from 1-10, not 0-9
    for i in range(sumArrCopy.__len__()):
        for j in range(sumArr.__len__()):
            if (sumArrCopy[i]==sumArr[j]):
                arrayRanking[i] += kArr[j]

    print(pStor, "sumArr", sumArr, file = f1)
    print(pStor, "maxCounts", maxCounts, file = f1)
    print(pStor, "arrRanking", arrayRanking, file = f1)
    print(pStor, "way2ranking", sortedWay2Arr, file = f1)
    print(pStor, "wPM10XVal", weightProbMaster, file = f1)
    print("\n", file = f1)
    f1.flush()
    ##### commented by Mama f1.close()


    #print("emptyClus", emptyClus)
    #print("clusCount", clusCount)



    trueClustering = [0]*10
    for i in range(masterDataSeqObj.__len__()):
        trueClustering[masterDataSeqObj[i].trueCluster]+=1

print("--- %s seconds ---" % (time.time() - start_time))
##### added by Mama
print("--- %s seconds ---" % (time.time() - start_time), file=f1)
f1.close()
#####



'''
alternative cross validation code
def ProbModel (probability):
    arrayProbs = [0]*K
    logArr = [0]*K
    xVal = seqOBJArr[3].value
    for k in range(K):
        prob = 0
        for x in range (xVal.__len__()):
            codon = xVal[x]
            prob += math.log10(probability[4*k + codon][x])
        gauss= (10**prob) * assignClusFreq[k]
        arrayProbs[k]= gauss
        logArr[k]= prob * assignClusFreq[k]
        print("prob", prob)
          #print("10^p", 10**prob)
    return(arrayProbs, logArr)


print(ProbModel(probability))
'''







