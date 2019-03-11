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
#from sequenceObj import binSearch
from sequenceObj import binSearch2

#go back to old probability func.?
#change trialFreq to 4, counter 1 default
trueK = 8
start_time = time.time()
# K = number of clusters
N = 91
# N = length of each sequence
S = 1415
# S = number of data points
printStor = 10
#number of times to run the whole thing, and store in a file!
f1= open("text.txt", "w+")
#opens the file to write in
tenPerc = (int)( S/10)
ninetyPerc=S-tenPerc
emptyClus = []
clusCount = []
numWarrCalc = 0

#to hold all the data from the file
masterDataSeqObj =[]

#number of times to run each cross validation (iot determine the best option)
bestOpRuns = 3

#convert to 5 fold, instead of 10 fold
#stop probability

kArr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
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
yep = open("/Users/mallika/PycharmProjects/DirichletBio/venv/lib/newflyData.txt", "r")
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

#functions for binary search are below:
def findMiddle(OBJarray, startPos, endPos):
    Len = endPos-startPos
    #("findMiddle: ", OBJarray)
    if (Len< 3):
        #print("too small")
        return
    else:
        mid= math.floor(Len/2)
        midPos = startPos + mid
        PosArr = [midPos-1, midPos, midPos+1]
        return (PosArr)


def bestOneArray(objArr):
    #find the position(s) with highest max counts

    #the following array contains the number of maxCounts per element
    maxCarr = [0]*objArr.__len__()
    #looks at each of the 10 array elements per positions and compares each pos across the positions
    for j in range (10):
        maxNum = -10000000000000000
        maxArrPos = -1
        for i in range(objArr.__len__()):
            if(objArr[i].wArr[j] > maxNum):
                maxNum = objArr[i].wArr[j]
                maxArrPos = i
            maxCarr[maxArrPos]+= 1

    #then, find the max in maxCarr
    max = 0
    for m in range(maxCarr.__len__()):
        if (maxCarr[m]> max):
            max = maxCarr[m]

    #then return all the objects that have that max maxCounts
    retval = []
    for n in range (maxCarr.__len__()):
        if (maxCarr[n]==max):
            retval.append(objArr[n])
    #print("retval", retval)
    return(retval)



def bestTwoArrays(arr1, arr2):
    #finds the best element in both return calls
    #print("in bestTwoArrays: ", arr1, ",", arr2)
    #find the bigger max
    max1 = bestOneArray(arr1)
    max2 = bestOneArray(arr2)
    #max1[0] to access the list of that one object
    #ie first has more maxCounts that second
    if (compare(max1[0].wArr, max2[0].wArr)==1):
        #print("returning from best2A: ", arr1)
        return (arr1)

    #ie second has more max counts
    if (compare(max1[0].wArr, max2[0].wArr)==-1):
        #print("returning from best2B: ", arr2)
        return (arr2)
    else:
        return (arr1 + arr2)



def binFunc(OBJarr, startPos, endPos):
    #base condition
    #print("binfunc", startPos, ",", endPos)
    if ((endPos-startPos) <= 3):
        #print("base case: <=4 left, ", startPos, ",", endPos)
        val= findwArr(OBJarr, startPos, endPos)
        retval = bestOneArray(val)
        return(retval)

    else:
        #returns the three middle in an arr
        PosArr = findMiddle(OBJarr, startPos, endPos)
        #print("middle = ", PosArr)
        result = findwArr(OBJarr, PosArr[0], PosArr[2])
        #print("values = ", result)
        #check if first > middle > last
        if ((compare(result[0].wArr, result[1].wArr) == 1) and compare(result[1].wArr, result[2].wArr)==1):
            endPos =  PosArr[0]
            #print("rec A binfunc", startPos, ",", endPos)
            return binFunc(OBJarr, startPos, endPos)

        #if the last > middle > first
        elif ((compare(result[2].wArr, result[1].wArr) == 1) and compare(result[1].wArr, result[0].wArr)==1):
            startPos = PosArr[2]
            #print("rec B binfunc", startPos, ",", endPos)
            return binFunc(OBJarr, startPos, endPos)

        #if middle is the highest
        elif ((compare(result[1].wArr, result[2].wArr) == 1) and compare(result[1].wArr, result[0].wArr)==1):
            #only want to find the sum of the middle
            retval = [result[1]]
            #print("bitfunc returning middle K = ", OBJarr[PosArr[1]].k, ", val = ", retval)
            return retval

        #if middle is lowest (need both directions), so go to the lese statement
        else:
            #print("middle is worst so calling both sides")
            #print("rec C binfunc", startPos, ",", PosArr[0], ">>", endPos)
            retval1 = binFunc(OBJarr, startPos,PosArr[0])
            #print("rec D binfunc", PosArr[2], ",", endPos, "<<", startPos)
            retval2 = binFunc(OBJarr, PosArr[2], endPos)
            return bestTwoArrays(retval1, retval2)




#checks the maxCounts for each of the 2 arrays
#returns -1 if first elem is has more maxcounts, 0 if they have an equal number, -1 if
def compare(arr1, arr2):
    mc1 = 0
    mc2= 0
    for i in range(10):
        if (arr1[i]>arr2[i]):
            mc1 +=1
        if (arr2[i]>arr1[i]):
            mc2+=1

    if (mc1 > mc2):
        return 1
    if (mc2 > mc1):
        return -1
    else: return 0



#instead of creating loops, finding the wPM sum is now transformed into a function, with a given K. Way2 methods etc are printed out

def findwArr(OBJarr, startPos, endPos):
    global numWarrCalc
    retObjArr = []
    pos = startPos
    #following code is pseudo
    while (pos <= endPos):
        K = OBJarr[pos].k
        #("Calculating Warr for K = ", K)
        if (OBJarr[pos].wArr[0]!=0):
                #print("wArr for k = ", K, " already exists, = ", OBJarr[pos].wArr)
                pos += 1
                retObjArr.append(OBJarr[pos])
                continue
        numWarrCalc += 1
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
                #print("bestOp", K, bestOp)
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

        #appends weightedProbAverage to get all 10 runs
        OBJarr[pos].wArr = weightProbAvg
        #print("wPA", weightProbAvg)
        retObjArr.append(OBJarr[pos])
        pos += 1
    return(retObjArr)

#ACTUAL FUNCTION CALL:

for p in range(printStor):
    numWarrCalc = 0
    #creting the kObject Array
    kObjArr = []
    for i in range(kArr.__len__()):
        obj = binSearch2.makebinSearch2(kArr[i], [0]*10)
        kObjArr.append(obj)

    bestRes = binFunc(kObjArr, 0, kObjArr.__len__()-1)
    print("bestResults: ", bestRes)
    print("Calculated wARR ", numWarrCalc, " / ", kObjArr.__len__(), " times", "run #", p+1)


print("--- %s seconds ---" % (time.time() - start_time))
print("actualTime", (time.time() - start_time)/printStor)











