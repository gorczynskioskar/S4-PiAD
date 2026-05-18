# Zadanie nr 1

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
Y = iris.target

print(f"X.shape: {X.shape}")
print(f"features: {iris.feature_names}")
print(f"classes: {np.unique(Y)}")

# Zadanie nr 2
from sklearn.cluster import AgglomerativeClustering
X_single = AgglomerativeClustering(n_clusters=3, linkage='single').fit_predict(X)
X_average = AgglomerativeClustering(n_clusters=3, linkage='average').fit_predict(X)
X_complete = AgglomerativeClustering(n_clusters=3, linkage='complete').fit_predict(X)
X_ward = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X)

print(f"single: {np.unique(X_single)}")
print(f"average: {np.unique(X_average)}")
print(f"complete: {np.unique(X_complete)}")
print(f"ward: {np.unique(X_ward)}")

# Metody aglomeracyjne działają hierarchicznie, łącząc stopniowo klastry
# Różne kryteria linkage określają, jak liczona jest odległość między klastrami

# Zadanie nr 3

import scipy.stats

def find_perm(clusters, Y_real, Y_pred):
    perm = []
    for i in range(clusters):
        idx = (Y_pred == i)
        new_label = scipy.stats.mode(Y_real[idx], keepdims=True)[0][0]
        perm.append(new_label)
    return np.array([perm[label] for label in Y_pred])

Y_ward_mapped = find_perm(3, Y, X_ward)

# Funkcja find_perm mapuje etykiety klastrów na rzeczywiste klasy,
# przypisując każdemu klastrowi etykietę klasy najczęściej w nim występującej.
# Proste porównanie Y_pred == Y_real może dawać błędne wyniki, ponieważ
# w klasteryzacji numery klastrów są umowne i mogą być dowolną permutacją prawdziwych klas.

# Zadanie nr 4

from sklearn.metrics import jaccard_score

jaccard = jaccard_score(Y, Y_ward_mapped, average='macro')

print(f"Jaccard (ward): {jaccard:.2f}")

# 0 - brak zgodności,
# 1 - idealne dopasowanie klastrów do klas,
# im wyższa wartość, tym lepsza jakość grupowania.

# Zadanie nr 5

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

def plot_hull(points):
    if len(points) >= 3:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0],
                     points[simplex, 1],
                     'k-', linewidth=1)

plt.figure(num = 'Zadanie nr 5', figsize=(21, 5))
plt.subplot(1,3,1)
for label in np.unique(Y):
    pts = X_reduced[Y == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Class {label}')
    plot_hull(pts)

plt.title('Rzeczywiste klasy')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)

plt.subplot(1,3,2)

for label in np.unique(Y_ward_mapped):
    pts = X_reduced[Y_ward_mapped == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {label}')
    plot_hull(pts)

plt.title('Klasy z klasteryzacji (Ward)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)


correct = (Y_ward_mapped == Y)

plt.subplot(1,3,3)
plt.scatter(
    X_reduced[correct, 0],
    X_reduced[correct, 1],
    c='green',
    label='Poprawnie sklasyfikowane',
    alpha=0.7
)

plt.scatter(
    X_reduced[~correct, 0],
    X_reduced[~correct, 1],
    c='red',
    label='Błędnie sklasyfikowane',
    alpha=0.7
)

plt.title('Poprawne vs błędne przypisania')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()

# Zadanie nr 6

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_reduced_3D = pca.fit_transform(X)

fig, axes = plt.subplots(
    1, 3,
    figsize=(24, 6),
    subplot_kw={'projection': '3d'},
    num = 'Zadanie nr 6'
)

# rzeczywiste klasy

ax = axes[0]
for label in np.unique(Y):
    pts = X_reduced_3D[Y == label]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               label=f'Class {label}', s=40)

ax.set_title('Rzeczywiste klasy (PCA 3D)')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.legend()

# klasteryzacja

ax = axes[1]
for label in np.unique(Y_ward_mapped):
    pts = X_reduced_3D[Y_ward_mapped == label]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               label=f'Cluster {label}', s=40)

ax.set_title('Klasteryzacja Ward (PCA 3D)')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.legend()

# poprawne vs błędne

correct = (Y_ward_mapped == Y)
ax = axes[2]

ax.scatter(X_reduced_3D[correct, 0],
           X_reduced_3D[correct, 1],
           X_reduced_3D[correct, 2],
           c='green', label='Poprawne', s=40)

ax.scatter(X_reduced_3D[~correct, 0],
           X_reduced_3D[~correct, 1],
           X_reduced_3D[~correct, 2],
           c='red', label='Błędne', s=40)

ax.set_title('Poprawne vs błędne przypisania (PCA 3D)')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.legend()

plt.tight_layout()
plt.show()

# Zadanie nr 7

from scipy.cluster import hierarchy

Z = hierarchy.linkage(X, method='ward')

plt.figure(num='Zadanie nr 7', figsize=(10, 6))

hierarchy.dendrogram(
    Z,
    truncate_mode='level',
    p=4
)

plt.title('Dendrogram (metoda Warda)')
plt.xlabel('Indeks próbek / klastrów')
plt.ylabel('Odległość (przyrost wariancji)')
plt.show()

# Oś pionowa dendrogramu przedstawia odległość lub przyrost wariancji,
# przy którym klastry są łączone. Liczbę klastrów odczytuje się, rysując
# poziomą linię na wybranej wysokości i zliczając liczbę przeciętych gałęzi.

# Zadanie nr 8

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

kmeans = KMeans(n_clusters=3, random_state=42)
Y_kmeans = kmeans.fit_predict(X)

Y_kmeans_mapped = find_perm(3, Y, Y_kmeans)

jaccard_kmeans = jaccard_score(Y, Y_kmeans_mapped, average='macro')

gmm = GaussianMixture(n_components=3, random_state=42)
Y_gmm = gmm.fit_predict(X)

Y_gmm_mapped = find_perm(3, Y, Y_gmm)

jaccard_gmm = jaccard_score(Y, Y_gmm_mapped, average='macro')

jaccard_ward = jaccard_score(Y, Y_ward_mapped, average='macro')

print("Porównanie metod (indeks Jaccarda):")
print(f"AgglomerativeClustering (ward): {jaccard_ward:.2f}")
print(f"KMeans:                       {jaccard_kmeans:.2f}")
print(f"GaussianMixture:              {jaccard_gmm:.2f}")

# ward
plt.figure(num='Zadanie nr 8', figsize=(21, 15))
plt.subplot(3,3,1)
for label in np.unique(Y):
    pts = X_reduced[Y == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Class {label}')
    plot_hull(pts)

plt.title('Rzeczywiste klasy')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)

plt.subplot(3,3,2)

for label in np.unique(Y_ward_mapped):
    pts = X_reduced[Y_ward_mapped == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {label}')
    plot_hull(pts)

plt.title('Klasy z klasteryzacji (Ward)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)


correct = (Y_ward_mapped == Y)

plt.subplot(3,3,3)

plt.scatter(
    X_reduced[correct, 0],
    X_reduced[correct, 1],
    c='green',
    label='Poprawnie sklasyfikowane',
    alpha=0.7
)

plt.scatter(
    X_reduced[~correct, 0],
    X_reduced[~correct, 1],
    c='red',
    label='Błędnie sklasyfikowane',
    alpha=0.7
)

plt.title('Poprawne vs błędne przypisania dla klasteryzacji ward')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)

# kmeans

plt.subplot(3,3,4)
for label in np.unique(Y):
    pts = X_reduced[Y == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Class {label}')
    plot_hull(pts)

plt.title('Rzeczywiste klasy')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)

plt.subplot(3,3,5)

for label in np.unique(Y_kmeans_mapped):
    pts = X_reduced[Y_kmeans_mapped == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {label}')
    plot_hull(pts)

plt.title('Klasy z klasteryzacji (KMeans)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)


correct = (Y_kmeans_mapped == Y)

plt.subplot(3,3,6)

plt.scatter(
    X_reduced[correct, 0],
    X_reduced[correct, 1],
    c='green',
    label='Poprawnie sklasyfikowane',
    alpha=0.7
)

plt.scatter(
    X_reduced[~correct, 0],
    X_reduced[~correct, 1],
    c='red',
    label='Błędnie sklasyfikowane',
    alpha=0.7
)

plt.title('Poprawne vs błędne przypisania z klasteryzacji kmeans')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)

# gmm

plt.subplot(3,3,7)
for label in np.unique(Y):
    pts = X_reduced[Y == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Class {label}')
    plot_hull(pts)

plt.title('Rzeczywiste klasy')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)

plt.subplot(3,3,8)

for label in np.unique(Y_gmm_mapped):
    pts = X_reduced[Y_gmm_mapped == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {label}')
    plot_hull(pts)

plt.title('Klasy z klasteryzacji (GMM)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)


correct = (Y_gmm_mapped == Y)

plt.subplot(3,3,9)

plt.scatter(
    X_reduced[correct, 0],
    X_reduced[correct, 1],
    c='green',
    label='Poprawnie sklasyfikowane',
    alpha=0.7
)

plt.scatter(
    X_reduced[~correct, 0],
    X_reduced[~correct, 1],
    c='red',
    label='Błędnie sklasyfikowane',
    alpha=0.7
)

plt.title('Poprawne vs błędne przypisania z klasteryzacji gmm')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()

# Zadanie nr 9

import pandas as pd

zoo = pd.read_csv('zoo.csv')

print(zoo.head())
print(zoo.columns)

X_zoo = zoo.drop(columns=['animal_name', 'type']).to_numpy()

Y_zoo = zoo['type'].to_numpy()

n_clusters = len(np.unique(Y_zoo))
print(f"Liczba klas w zbiorze zoo: {n_clusters}")

zoo_ward = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward'
)

Y_zoo_pred = zoo_ward.fit_predict(X_zoo)

Y_zoo_mapped = find_perm(n_clusters, Y_zoo, Y_zoo_pred)

jaccard_zoo = jaccard_score(Y_zoo, Y_zoo_mapped, average='macro')
print(f"Jaccard (zoo, ward): {jaccard_zoo:.2f}")

# Indeks Jaccarda dla zoo jest niższy niż dla zbioru iris,
# co wynika z bardziej złożonej struktury danych.

pca_zoo = PCA(n_components=2)
X_zoo_reduced = pca_zoo.fit_transform(X_zoo)

plt.figure(num='Zadanie nr 9', figsize=(21, 5))
plt.subplot(1,3,1)

for label in np.unique(Y_zoo):
    pts = X_zoo_reduced[Y_zoo == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Class {label}')
    plot_hull(pts)

plt.title('Rzeczywiste klasy (zoo, PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)

plt.subplot(1,3,2)

for label in np.unique(Y_zoo_mapped):
    pts = X_zoo_reduced[Y_zoo_mapped == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {label}')
    plot_hull(pts)

plt.title('Klasteryzacja zoo (Ward, PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)

correct = (Y_zoo_mapped == Y_zoo)

plt.subplot(1,3,3)

plt.scatter(
    X_zoo_reduced[correct, 0],
    X_zoo_reduced[correct, 1],
    c='green',
    label='Poprawne',
    alpha=0.7
)

plt.scatter(
    X_zoo_reduced[~correct, 0],
    X_zoo_reduced[~correct, 1],
    c='red',
    label='Błędne',
    alpha=0.7
)

plt.title('Poprawne vs błędne przypisania (zoo)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()

# Zadanie nr 10

from sklearn import datasets
from sklearn.metrics import calinski_harabasz_score

iris = datasets.load_iris()
X = iris.data

k_values = range(2, 9)

inertias = []
ch_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    inertias.append(kmeans.inertia_)

    ch = calinski_harabasz_score(X, labels)
    ch_scores.append(ch)

plt.figure(num='Zadanie nr 10', figsize=(14, 5))
plt.subplot(1,2,1)
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Liczba klastrów k')
plt.ylabel('Inercja')
plt.title('Metoda łokcia (k-means, iris)')
plt.grid(True, alpha=0.4)

plt.subplot(1,2,2)
plt.plot(k_values, ch_scores, marker='o')
plt.xlabel('Liczba klastrów k')
plt.ylabel('Indeks Calińskiego–Harabasza')
plt.title('Indeks Calińskiego–Harabasza (iris)')
plt.grid(True, alpha=0.4)
plt.show()

# Na podstawie metody łokcia oraz indeksu Calińskiego–Harabasza optymalna
# liczba klastrów dla zbioru iris wynosi 3. Jest to zgodne z rzeczywistą
# liczbą klas w danych.

# Poza inercją i indeksem Calińskiego–Harabasza można wykorzystać m.in. indeks
# Sylwetki, indeks Daviesa–Bouldina, kryteria informacyjne (AIC, BIC)
# lub analizę stabilności klastrów.