# Zadanie nr 1

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

def distE(X, C):
  d=np.linalg.norm(X[:, None, :] - C, axis=-1)
  return d

X=np.array([[0,0],[3,4],[1,1]])
C=np.array([[0,0],[3,4]])
print(distE(X, C))

# Zadanie nr 2

def distM(X, C):
  V = np.cov(X, rowvar=False)
  V_inv = np.linalg.inv(V)

  diff = X[:, None, :] - C

  d = np.sqrt(np.sum((diff @ V_inv) * diff, axis=-1))

  return d

print(distM(X, C))

# Odległość Mahalanobisa uwzględnia kowariancję danych,
# dzięki czemu skaluje i „obraca” przestrzeń cech, podczas gdy odległość
# euklidesowa traktuje wszystkie wymiary jednakowo. Jest szczególnie przydatna,
# gdy cechy mają różne wariancje lub są ze sobą skorelowane.

# Zadanie nr 3

def ksrodki(X, k, dist_fn=distE):
  n, m = X.shape
  indices = np.random.choice(n, k, replace=False)
  C = X[indices]
  labels = np.zeros(n)

  while True:
    D = dist_fn(X, C)
    new_labels = np.argmin(D, axis=1)

    if np.array_equal(labels, new_labels):
      break

    labels = new_labels

    for j in range(k):
        if np.any(labels == j):
            C[j] = X[labels == j].mean(axis=0)

  return labels, C

labels, C = ksrodki(X, 2)

def F_C(X, labels, C, dist_fn):
    K = C.shape[0]

    num = 0
    for k in range(K):
        for l in range(k + 1, K):
            num += np.linalg.norm(C[k] - C[l])

    D = dist_fn(X, C)
    denom = 0
    for i in range(len(X)):
        denom += D[i, labels[i]]**2

    return num / denom

# Duże F(C) - dobra klasteryzacja

print("labels:", labels)
print("centroids:\n", C)

# Zadanie nr 4

import pandas as pd

df = pd.read_csv('autos.csv', na_values='?')

X = df.select_dtypes(include='number').dropna()

print("Wymiary macierzy X:", X.shape)
print("Kolumny:\n", X.columns)

# Zadanie nr 5

k = 3
labels, C = ksrodki(X.values, k)

print("Liczność klastrów:")
for i in range(k):
    print(f"Klaster {i}: {np.sum(labels == i)}")

fc = F_C(X.values, labels, C, distE)
print(f"\nF(C): {fc:.10f}")

print("\nCentroidy (pierwsze 3 kolumny):")
print(C[:, :3])

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.values)

plt.figure(num="Zadanie nr 5", figsize=(7,5))

for i in range(k):
    pts = X_reduced[labels == i]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Klaster {i}')

C_reduced = pca.transform(C)

plt.scatter(
    C_reduced[:, 0],
    C_reduced[:, 1],
    c='black',
    marker='x',
    s=100,
    label='Centroidy'
)

plt.title('k-means: autos (PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.show()

# Zadanie nr 6

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
Y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3
labels, C = ksrodki(X_scaled, k)

print("Liczność klastrów:")
for i in range(k):
    print(f"Klaster {i}: {np.sum(labels == i)}")

fc = F_C(X_scaled, labels, C, distE)
print(f"\nF(C): {fc:.6f}")

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

plt.figure(num="Zadanie nr 6", figsize=(7,5))

for i in range(k):
    pts = X_reduced[labels == i]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Klaster {i}')

C_reduced = pca.transform(C)
plt.scatter(C_reduced[:, 0], C_reduced[:, 1],
            c='black', marker='x', s=100, label='Centroidy')

plt.title('k-means: iris (PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.show()

# Zadanie nr 7

# Euklides
labels_E, C_E = ksrodki(X_scaled, 3, dist_fn=distE)
fc_E = F_C(X_scaled, labels_E, C_E, distE)
print(f"Euklides: F(C) = {fc_E:.6f}")

# Mahalanobis
labels_M, C_M = ksrodki(X_scaled, 3, dist_fn=distM)
fc_M = F_C(X_scaled, labels_M, C_M, distM)
print(f"Mahalanobis: F(C) = {fc_M:.6f}")

# Na zbiorze iris lepsze wyniki daje zwykle odległość euklidesowa,
# ponieważ dane mają podobne skale i niewielkie korelacje między cechami.
# Odległość Mahalanobisa uwzględnia kowariancję, co jest korzystne głównie
# przy silnie skorelowanych lub różnoskalowych danych.

# Zadanie nr 8

import scipy.stats

def find_perm(clusters, Y_real, Y_pred):
    perm = []
    for i in range(clusters):
        idx = (Y_pred == i)
        new_label = scipy.stats.mode(Y_real[idx], keepdims=True)[0][0]
        perm.append(new_label)
    return np.array([perm[label] for label in Y_pred])

from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score

# sklearn KMeans
km = KMeans(n_clusters=3, random_state=42)
labels_sk = km.fit_predict(X)

# mapowanie etykiet
labels_sk_mapped = find_perm(3, Y, labels_sk)
labels_my_mapped = find_perm(3, Y, labels)

# Jaccard
jaccard_my = jaccard_score(Y, labels_my_mapped, average='macro')
jaccard_sk = jaccard_score(Y, labels_sk_mapped, average='macro')

# F(C)
fc_my = F_C(X_scaled, labels, C, distE)

# tabela wyników
print("         | Jaccard | F(C)")
print("--------------------------")
print(f"ksrodki  | {jaccard_my:.2f}   | {fc_my:.4f}")
print(f"KMeans   | {jaccard_sk:.2f}   | --")

# Zadanie nr 9

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

X_data = df.select_dtypes(include='number').dropna().values

# skalowanie
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# ksrodki
labels_my, C_my = ksrodki(X_scaled, 3)

# sklearn
km = KMeans(n_clusters=3, random_state=42)
labels_sk = km.fit_predict(X_scaled)

# CH score
ch_my = calinski_harabasz_score(X_scaled, labels_my)
ch_sk = calinski_harabasz_score(X_scaled, labels_sk)

# F(C)
fc_my = F_C(X_scaled, labels_my, C_my, distE)

print("         | CH score | F(C)")
print("----------------------------")
print(f"ksrodki  | {ch_my:.2f}    | {fc_my:.4f}")
print(f"KMeans   | {ch_sk:.2f}    | --")

# Wartości indeksu Calińskiego–Harabasza dla własnej implementacji oraz
# algorytmu sklearn są bardzo zbliżone, co potwierdza poprawność działania
# zaimplementowanego algorytmu k‑środków. Niewielkie różnice wynikają z losowej
# inicjalizacji centroidów oraz zastosowania wielokrotnych inicjalizacji
# i optymalizacji w bibliotece sklearn.

# Zadanie nr 10

k_values = range(2, 9)
F_values = []

for k in k_values:
    best_fc = -np.inf

    for _ in range(10):
        labels_tmp, C_tmp = ksrodki(X_scaled, k)
        fc_tmp = F_C(X_scaled, labels_tmp, C_tmp, distE)

        if fc_tmp > best_fc:
            best_fc = fc_tmp

    F_values.append(best_fc)

plt.figure(num="Zadanie nr 10", figsize=(7,5))
plt.plot(k_values, F_values, marker='o')

plt.xlabel('Liczba klastrów k')
plt.ylabel('F(C)')
plt.title('F(C) w zależności od k')
plt.grid(True)
plt.show()

# Wartość funkcji F(C) rośnie wraz ze wzrostem liczby klastrów,
# ponieważ punkty znajdują się coraz bliżej swoich centroidów.
# Oznacza to, że miara F(C) nie jest odpowiednia do wyboru optymalnego k,
# gdyż preferuje większą liczbę klastrów.

# Wyniki algorytmu k‑środków mogą się różnić między uruchomieniami,
# ponieważ centroidy są inicjalizowane losowo. W praktyce problem ten
# rozwiązuje się przez wielokrotne uruchamianie algorytmu i wybór najlepszego
# wyniku lub ustawienie random_state i użycie wielu inicjalizacji
# (np. w sklearn n_init).