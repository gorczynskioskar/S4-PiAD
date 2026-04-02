import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

# Zadanie nr 1

print("Zadanie nr 1")

def freq(x, prob=True):
    values=[]
    prob_count=np.zeros(max(x))

    for n in x:
        prob_count[n-1]+=1
        if n not in values:
            values.append(n)
    
    if prob:
        prob_count/=len(x)
    
    return [values, prob_count]

x = [1,1,2,3,3,3]
(val, prob) = freq(x, True)
(val, count) = freq(x, False)
print(f"Dane wejsciowe:\tx = {x}\nWynik:\nwartosci:\t\t{val}\nlicznosci:\t\t{count}\nprawdopodobienstwa:\t{prob}")

# Zadanie nr 2

print("Zadanie nr 2")

def freq2(x, y, prob=True):
    pairs=[]
    prob_count=[]

    for i in range(prob_count.size()):
        prob_count()

        pair=[x[i], y[i]]
        if pair not in pairs:
            pairs.append(pair)

# Zadanie nr 3
            
print("Zadanie nr 3")

def entropy(x):
    H_sum=0
    probs=np.zeros(max(x)+1)
    for n in x:
        probs[n]+=1
    
    probs/=len(x)

    for n in probs:
        if n!=0:
            H_sum-=n*np.log2(n)

    return (H_sum)


print(entropy([1,1,1,1]))
print(entropy([0,1]))

# Zadanie nr 4

print("Zadanie nr 4")

def infogain(x,y):
    H_x=entropy(x)
    H_y=entropy(y)

# Zadanie nr 5
    
print("Zadanie nr 5")

zoo=pd.read_csv('zoo.csv')
print(zoo.head())
print(zoo.columns)