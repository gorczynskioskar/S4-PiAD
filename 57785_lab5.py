import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.datasets import fetch_rcv1
import time

# Zadanie nr 1

print("Zadanie nr 1")


def freq(x, prob=True):

    if sp.issparse(x):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()

    values, counts = np.unique(x, return_counts=True)

    if prob:
        counts = counts / len(x)

    return values, counts


x = [1,1,2,3,3,3]
(val, prob) = freq(x, True)
(val, count) = freq(x, False)
print(f"Dane wejsciowe:\tx = {x}\nWynik:\nwartosci:\t\t{val}\nlicznosci:\t\t{count}\nprawdopodobienstwa:\t{prob}")

# Zadanie nr 2

print("Zadanie nr 2")

def freq2(x, y, prob=True):

    if sp.issparse(x):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()

    if sp.issparse(y):
        y = y.toarray().ravel()
    else:
        y = np.asarray(y).ravel()

    pairs = np.column_stack((x, y)).astype(int)

    values, counts = np.unique(pairs, axis=0, return_counts=True)

    if prob:
        counts = counts / len(x)

    return values, counts

x=[0,0,1,1]
y=[1,1,0,1]

values, prob_count = freq2(x, y, True)
for n in range(len(values)):
    print(f"{values[n]}: {prob_count[n]}")

# Zadanie nr 3
            
print("Zadanie nr 3")

def entropy(x):
    H_sum=0
    (values, probs)=freq(x, True)

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
    
    values, probs = freq2(x, y, prob=True)
    H_xy = 0
    for p in probs:
        if p > 0:
            H_xy -= p * np.log2(p)

    return H_x + H_y - H_xy

x=[0,0,1,1]
y=[1,1,0,1]

print(infogain(x,x))
print(infogain(x,y))

# Zadanie nr 5
    
print("Zadanie nr 5")

zoo=pd.read_csv('zoo.csv')
print(zoo.head())
print(zoo.columns)

# Zadanie nr 6
    
print("Zadanie nr 6")

df6 = zoo.drop(columns=['animal_name', 'type'])
df6 = df6.astype(int)
Y = zoo['type'].astype(int).to_numpy()
infogain_df = pd.DataFrame(columns=df6.columns)
for col in df6.columns:
    X = df6[col].to_numpy()
    infogain_df.loc[0, col] = infogain(X, Y)

infogain_df = infogain_df.transpose().sort_values(by=0, ascending=False)

# Zadanie nr 7

print("Zadanie nr 7")

plt.figure()
plt.bar(infogain_df.head(10).index, infogain_df.head(10)[0])
plt.xticks(rotation=90)
plt.xlabel('Cecha')
plt.ylabel('I(Cecha; typ)')
plt.title('Wzrost informacji o typie zwierzęcia w zależności od cechy')
plt.show()

# Zadanie nr 8

print("Zadanie nr 8")

X_sparse = sp.csr_matrix([[1,0,3],[0,2,0],[2,0,0]])

freq(X_sparse[0])
freq2(X_sparse[0], X_sparse[1])

freq2(X_sparse[:,0], X_sparse[:,1])

# Zadanie nr 9

print("Zadanie nr 9")

rcv1 = fetch_rcv1(data_home="C:\\Users\\gorcz\\scikit_learn_data")
X = rcv1.data
Y = rcv1.target[:, 87]

X_bin = (X > 0).astype(int)

X_csc = X_bin.tocsc()

Y_vec = Y.toarray().ravel()

n_features = X_csc.shape[1]
N = X_csc.shape[0]

info_gains = []

start = time.time()

for j in range(1000): # ograniczenie do 1000 cech ze względu na czas wykonania
    col = X_csc[:, j]
    x_feat = col.toarray().ravel()

    ig = infogain(x_feat, Y_vec)
    info_gains.append((j, ig))

    print(f"Przetwarzanie: {j}")

print("Czas wykonania:", time.time() - start, "s")

info_gains.sort(key=lambda x: x[1], reverse=True)

top50 = info_gains[:50]

print("\nTOP 50 cech:")
for idx, ig in top50:
    print(f"word_{idx}: {ig}")