#18EE10068
#Shirsha Chowdhury
#Assignment 1:Decision Trees

#For data input after adding the path name add r in the begining of the name


import pandas as pd   #pandas is used to read the values from the csv files
import math

def Entropy(p):         #Shanon's formula to calculate Entropy
    if (p==0 or p==1):
        return 0
    return -(p*math.log2(p))

def sumArr(arr):      #function to calculate array sum
    s=0
    for x in arr:
        s+=x
    return s

def sumEntropy(d,N):
    s=0
    for key in d:
        temp=0
        cnt=sumArr(d[key])
        for val in d[key]:
            probability=(val*1.0/cnt)
            temp+=Entropy(probability)
        s+=((sumArr(d[key])*1.0)/N)*temp
    return s

def totalEntropy(df):      #Calculates the total Entropy in the target variable of the data
    val,N=0,len(df)
    d={}
    for row in df['target']:
        if row not in d:
            d[row] = 0
        d[row]+=1
    for key in d:
        probability=(d[key]*1.0)/N
        val+=Entropy(probability)
    return val

def bestSplit(df):   #Returns the attribute having maximum information gain
    encode={}        #Encoding Different Class as 0,1,2,3
    c=0
    for row in df['target']:
        if row not in encode:
            encode[row]=c
            c+=1
    numClass=len(encode)
    infoGainMax=0
    attri='None'
    E=totalEntropy(df)
    N=len(df)
    for i in range(len(df.columns)-1):
        ''' cat here stores the count of different target class for each
            class in the attribute of a feature. This is iterated for all the features
            to know the information gain of that particular feature'''
        cat={}           
        for row in df[df.columns[i]]:
            if row not in cat:
                cat[row]=[0]*numClass
        it=0
        for row in df[df.columns[i]]:
            cat[row][encode[df['target'][it]]]+=1
            it+=1
        IG=E-sumEntropy(cat,N)  #Checking the Maximum Information Gain
        if IG>=infoGainMax:
            infoGainMax=IG
            attri=df.columns[i]
    return attri

def base1(df):
    
    ''' First Base Case is to check if all the target values
        of a dataset belong to the same class. If that happens we 
        terminate by creating the leaf node with value as that of the 
        target class '''

    val=df['target'][0]
    flag=0
    for x in df['target']:
        if (x!=val):
            flag=1
            break
    if flag==0:
        return val
    return 0

def base2(df):
    
    ''' Second Base Case is used when all the attributes have been used 
        and the value of leaf node will be determined by the majority class 
        present in the target column'''
    
    d={}
    for row in df['target']:
        if row not in d:
            d[row] = 0
        d[row]+=1
    attri,mx='null',0
    for key in d:
        if (d[key]>mx):
            mx=d[key]
            attri=key
    return attri

def trainTree(df):    #Main function which builds the tree recursively

    ''' First base1 and base2 function is called to check if a termination case 
        is encountered, in that case the branch is terminated with the class value. 
        Else bestSplit function is called to get the attribute with highest Information Gain. 
        After that new datasets are created by splitting the tree according to the best attribute. 
        The new dataset is again passed onto the trainTree function and the retuned tree will 
        be the subtree of the original tree'''
    
    totalAttributes=len(df.columns)-1
    if (base1(df)):
        return base1(df)
    if (totalAttributes==0):
        return base2(df)
    optimalAttribute=bestSplit(df)
    cat=[]
    for x in df[optimalAttribute]:
        if x not in cat:
            cat.append(x)
    tree={}
    for x in cat:
        ''' New dataset named as newdf is created by splitting the optimal 
            attribute by it's classes'''
        newdf=df[df[optimalAttribute]==x]         
        newdf=newdf.drop(optimalAttribute,axis=1) #The optimal attribute is dropped from the new dataset
        question= optimalAttribute+"="+str(x)
        newdf.index = range(len(newdf))
        child=trainTree(newdf)
        tree[question]=child
    return tree

def predictRow(arr,tree): #functin to predict the value for each test example
    if (type(tree)==str):
        return tree
    for x in arr:
        if x in tree:
            new_tree=tree[x]
            return predictRow(arr,new_tree)
        
def predict(test): #This function returns the prediction array
    result=[]
    for i in test.index:
        encode=[]
        for x in test.columns:
            temp=x+'='+str(test[x][i])
            encode.append(temp)
        result.append(predictRow(encode,tree))
    return result

def accuracy(result,test_target):  #function to calculate the accuracy
    #accuracy is being calculated as total correct classified/ target set size
    n,c=len(result),0
    for i in range(len(res)):
        if (result[i]==test_target[i]):
            c+=1
    return (c*1.0)/n

def printTree(d,tree): #funciton to print the tree 
    for key in tree:
        if (type(tree[key])==str):
            print('---'*d,end="|")
            print(key+':'+tree[key])
        else:
            print('---'*d,end="|")
            print(key)
            printTree(d+1,tree[key])

data = pd.read_csv(r"C:\Users\HP\Downloads\project1.data",header=None) #training data is loaded 
data.columns = ['price', 'maint', 'doors', 'persons','lug_boot','safety','target']

test = pd.read_csv(r"C:\Users\HP\Downloads\project1_test.data",header=None) #test data is loaded
test.columns = ['price', 'maint', 'doors', 'persons','lug_boot','safety','target']

#In the test set some values of doors were equal to 56, that was changed to 6.
test['doors']=[6 if x==56 else x for x in test['doors']]

test_target=test['target']  #Expected target column
test=test.drop('target',axis=1)

tree=trainTree(data)  
res=predict(test)
print('The final tree is:')
print()
printTree(0,tree)
print()
accu=(accuracy(res,test_target))*100
print("Accuracy of the test set is "+str(accu)+"%")