#Roll:18EE10068
#Name:Shirsha Chowdhury
#Machine Learning: Project 3: K-Means Clustering


import pandas as pd
import numpy as np
import random
import math

data = pd.read_csv(r"C:\Users\HP\Downloads\Project3.csv")
sample=data.copy()

m=len(data)
n=len(data.columns)


def Zscore(arr):                    #Function which replaces the attribute with Zscore 
    x=np.array(arr)
    u=np.mean(arr)
    s=np.std(arr)
    den=np.subtract(x,u)
    z=np.divide(den,s)
    return z

def distance(A,B):       #To find the distance between two data points (i.e Two rows of the dataset)
    quant=[3,5]
    dist=0
    for i in range(1,len(A)):
        if (i in quant):
            dist+=(A[i]-B[i])**2
        else:
            dist+=(A[i]!=B[i])
    return math.sqrt(dist)

def Mean(arr):            #Finding the centroid of a Cluster

    """For quantitative data mean was taken and for categorical data mode will be taken"""
    quant=[3,5]
    ans=[0 for i in range(8)]
    N=len(arr)
    if (N==0):
        return ans
    for i in range(1,n):
        if (i in quant):
            total=0
            for c in arr:
                total+=c[i]
            ans[i]=total/N
        else:
            v=[]
            for c in arr:
                v.append(c[i])
            ans[i]=np.bincount(v).argmax()
    return ans

def randomInit(K):   #Randomly Initializes K data points as centroid
    Centroids=[]
    X = sample.iloc[:].values
    randomPoints=[]
    while (len(randomPoints)<K):
        rand=random.randint(0,m-1)
        if rand not in randomPoints:
            randomPoints.append(rand)
    for i in randomPoints:
        Centroids.append(X[i])
    return Centroids


def KMeans(n_iter,K):   
    """ We pass n_iter and K in this function and it returns the final 
    cluster which can after n_iter iterations. It was seen that the Cluster were getting converged withing 
    10 iterations"""
    
    Cluster={}   
    """Dictinary in which the keys are the cluster number and values are the list of 
        data point belonging to that cluster"""
    Centroids=randomInit(K)
    for it in range(n_iter):
        for i in range(K):
            Cluster[i+1]=[]
        for i in range(m):
            trainRow=sample.iloc[i]
            D=[]
            for C in Centroids:
                dist=distance(trainRow,C)
                D.append(dist)
            clusterNo=np.argmin(D)+1
            Cluster[clusterNo].append(trainRow)
        for key in Cluster:
            Centroids[key-1]=Mean(Cluster[key])
    return Cluster

        
def MSE(Cluster): #Returns the MSE of our distribution
    """This function was used to see the Error vs K plot from which the optimized K
    was taken"""
    dist=0
    for k in Cluster:
        centroid=Mean(Cluster[k])
        for dpoint in Cluster[k]:
            dist+=distance(centroid,dpoint)**2
    return dist/2000

def dataPrint(Cluster): #Prints the output in a file
    d={}
    file=open(r"C:\Users\HP\Downloads\18EE10068_P3.out","w")   #Writing the file
    for key in Cluster:
        for dpoint in Cluster[key]:
            d[dpoint.ID]=key
    for i in range(len(sample)):
        file.write(str(d[sample.iloc[i].ID])+" ")
    file.close()

def printCluster(Cluster): #Prints the distribution of points in different Clusters
    for key in Cluster:
        print(len(Cluster[key]),end=" | ")
    print()

sample['Age']=Zscore(sample['Age'])
sample['Income']=Zscore(sample['Income'])

number_of_iterations=20
optimized_K=15
Cluster=KMeans(number_of_iterations,optimized_K)
dataPrint(Cluster)

