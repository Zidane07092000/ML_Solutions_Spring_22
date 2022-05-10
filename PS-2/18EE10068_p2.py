#Roll:18EE10068
#Name:Shirsha Chowdhury
#Machine Learning Assignment 2: K-NN Classifier

import pandas as pd
import numpy as np
import math


VDM={}  # To store all the VDMs of discrete values

def Zscores(arr): # To find the Z scores of quantitative data
    x=np.array(arr)
    u=np.mean(arr)
    s=np.std(arr)
    den=np.subtract(x,u)
    z=np.divide(den,s)
    return z

def vdm(train,a,x,y): # To find the Value Distance Metric of Nominal data
    c=data['target'].unique()
    dist=0
    if (x==y):
        return 0
    for cl in c:
        n_axc,n_ax,n_ayc,n_ay=0,0,0,0
        for i in range(len(train[a])) :
            if (train[a][i]==x):
                n_ax+=1
            if (train['target'][i]==cl and train[a][i]==x):
                n_axc+=1
            if (train[a][i]==y):
                n_ay+=1
            if (train['target'][i]==cl and train[a][i]==y):
                n_ayc+=1
        dist+=((n_axc/n_ax)-(n_ayc/n_ay))**2
    return dist

def distance(sample,train, test):  #Finds the distance between 2 vectors
    quant=[0,3,4,7,9]
    dist=0
    for i in range(len(train)):
        if (i in quant):
            dist+=(train[i]-test[i])**2
        else:
            l=[train[i],test[i]]
            l.sort()
            tup=(*l,)
            if (i,tup) in VDM:
                dist+=VDM[i,tup]**2
            else:
                VDM[i,tup]=vdm(sample,sample.columns[i],train[i],test[i])
                dist+=VDM[i,tup]**2
    return math.sqrt(dist)

def getDistance(sample,test):   
    ''' Gets the distance vector for all the distances between the 
    training examples and the single test example'''
    distances=[]
    dist=0
    for i in sample.index:
        trainRow=[]
        for x in sample.columns:
            if (x!='target'):
                trainRow.append(sample[x][i])
        dist=distance(sample,trainRow, test)
        distances.append((i, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances

def getNeighbors(dist, num_neighbors): 
    '''Function which returns the k nearest neighbors according
        to the distance array'''
        
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(dist[i][0])
    return neighbors

def majorityVoting(data,arr):
    
    ''' This function takes in the neighbor array and returns the
        majority class in present in the neighbor array'''
    p,N=0,len(arr)
    for i in arr:
        if (data['target'][i]):
            p+=1
    if (p>N-p):
        return 1
    return 0

def Output(sample,test,k):
    '''function to print the optimal k and 
    user defined k. The optimal K came out to be at 
    k=19, which we obtained by dividing the training set into 
    2 parts in the (training set and validation set) in the ration 0.85:0.15
    and got the highest accuracuy of 86.67% at k=19'''
    '''Also in general the optimal value of K lies in between the value of 
    root(N), where N is the number of training examples.'''
    
    file=open(r"C:\Users\HP\Downloads\18EE10068_P2.out","w")
    file.write("Output for k="+str(k)+": ")
    for i in test.index:
        testRow=[]
        for x in test.columns:
            testRow.append(test[x][i])
        D=getDistance(sample,testRow)
        neighbours=getNeighbors(D,k)
        file.write(str(majorityVoting(sample,neighbours))+" ")
    file.write('\n')
    file.write("The Output for optimum K (K=19):")
    for i in test.index:
        testRow=[]
        for x in test.columns:
            testRow.append(test[x][i])
        D=getDistance(sample,testRow)
        neighbours=getNeighbors(D,19)
        file.write(str(majorityVoting(sample,neighbours))+" ")
    file.close()



data = pd.read_csv(r"C:\Users\HP\Downloads\project2.csv") #Reading the training data
test = pd.read_csv(r"C:\Users\HP\Downloads\project2_test.csv") #Reading the test data

d={}
quant=[0,3,4,7,9]   #index for quantitative data
for i in quant:
    d[i]=(np.mean(data[data.columns[i]]),np.std(data[data.columns[i]]))

sample=data.copy()

sample['chol']=Zscores(sample['chol'])
sample['trestbps']=Zscores(sample['trestbps'])
sample['age']=Zscores(sample['age'])
sample['thalach']=Zscores(sample['thalach'])
sample['oldpeak']=Zscores(sample['oldpeak'])


#Normalizing the test data set using Z score
for i in quant:          
    x=np.array(test[test.columns[i]])
    den=np.subtract(x,d[i][0])
    z=np.divide(den,d[i][1])
    test[test.columns[i]]=z


k=int(input("Enter Value of K:") )
Output(sample,test,k)











