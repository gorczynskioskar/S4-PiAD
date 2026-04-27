import numpy as np

# Zadanie nr 1

def wiPCA(X, n_components):

    mean_vector = np.mean(X, axis=0)
    X_centered = X - mean_vector
    cov_mat = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvector_subset = eigenvectors[:, :n_components]

    X_reduced = X_centered @ eigenvector_subset

    return X_reduced, eigenvector_subset, eigenvalues, mean_vector

import matplotlib.pyplot as plt

# Zadanie nr 2

np.random.seed(42)
A = np.random.randn(2, 2)
X_2d = np.random.randn(200, 2) @ A

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7, color='teal')
plt.title('Zadanie 2: Chmura 200 skorelowanych punktów')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Zadanie nr 3

# redukcja do jednego wymiaru
X_red, eigvecs, eigvals, mean_vec = wiPCA(X_2d, n_components=1)

# pierwszy wektor własny
v1 = eigvecs[:, 0]

# Rekonstrukcja rzutów w przestrzeni 2D
X_projected = mean_vec + np.outer(X_red[:, 0], v1)

plt.figure(figsize=(10, 8))

# oryginalne punkty
plt.scatter(X_2d[:, 0], X_2d[:, 1], color='green', label='Oryginalne punkty')

# rzuty na pierwszą składową
plt.scatter(X_projected[:, 0], X_projected[:, 1],
            color='red', label='Rzuty na 1. składową')


# pierwszy wektor własny jako strzałka
scale = 3  # długość strzałki
plt.quiver(mean_vec[0], mean_vec[1],
           v1[0] * scale, v1[1] * scale,
           angles='xy', scale_units='xy', scale=1,
           width=0.005, color='black', label='1. wektor własny')

plt.title('Zadanie 3: PCA – rzutowanie na pierwszą składową')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.show()

# Zadanie nr 4

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data          # macierz cech (150 x 4)
y = iris.target        # etykiety klas (0, 1, 2)

print("Rozmiar macierzy danych:", X.shape)
print("Nazwy cech:", iris.feature_names)
print("Nazwy klas:", iris.target_names)

# Zadanie nr 5

# redukcja do 2 wymiarów
X_red, eigvecs, eigvals, mean_vec = wiPCA(X, n_components=2)

plt.figure(figsize=(8, 6))

for class_label, class_name in enumerate(iris.target_names):
    plt.scatter(
        X_red[y == class_label, 0],
        X_red[y == class_label, 1],
        label=class_name,
        alpha=0.7
    )

plt.xlabel('1. składowa główna')
plt.ylabel('2. składowa główna')
plt.title('Zadanie 5: PCA (2D) dla zbioru iris')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.show()

# Zadanie nr 6

# wczytanie zbioru digits
digits = datasets.load_digits()
X = digits.data      # (1797 x 64)
y = digits.target    # etykiety 0–9

# redukcja do 2 wymiarów
X_red, eigvecs, eigvals, mean_vec = wiPCA(X, n_components=2)

# Wizualizacja
plt.figure(figsize=(8, 7))

scatter = plt.scatter(
    X_red[:, 0],
    X_red[:, 1],
    c=y,
    cmap='tab10',
    alpha=0.7,
    s=15
)

plt.xlabel('1. składowa główna')
plt.ylabel('2. składowa główna')
plt.title('Zadanie 6: PCA (2D) dla zbioru digits')
plt.colorbar(scatter, label='Cyfra')
plt.grid(True, linestyle='--', alpha=0.4)
plt.axis('equal')
plt.show()

# Zadanie nr 7

# PCA na pełnym wymiarze (64 składowe)
X_red, eigvecs, eigvals, mean_vec = wiPCA(X, n_components=X.shape[1])

# udział wariancji wyjaśnianej przez każdą składową
explained_variance_ratio = eigvals / np.sum(eigvals)

# skumulowana wariancja
cumulative_variance = np.cumsum(explained_variance_ratio)

# wykres
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker='o'
)

plt.xlabel('Liczba składowych głównych')
plt.ylabel('Skumulowany udział wariancji')
plt.title('Zadanie 7: Skumulowana wariancja – PCA (digits)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, 1.01)
plt.show()

# Liczba składowych potrzebna do 90% wariancji
k_90 = np.argmax(cumulative_variance >= 0.9) + 1
print(f"Liczba składowych potrzebna do zachowania ≥ 90% wariancji: {k_90}")

# Zadanie nr 8

# wczytanie zbioru iris
iris = datasets.load_iris()
X = iris.data

# własna implementacja PCA
X_wi, eigvecs, eigvals, mean_vec = wiPCA(X, n_components=2)

# PCA ze sklearn
pca_sk = PCA(n_components=2)
X_sk = pca_sk.fit_transform(X)

# porównanie (z dokładnością do znaku wektorów własnych)
comparison = np.allclose(
    np.abs(X_wi),
    np.abs(X_sk),
    atol=1e-6
)

print("Czy wyniki wiPCA i sklearn.PCA są zgodne (do znaku)?", comparison)

# Zadanie nr 9

def wiPCA_inverse(X_reduced, eigenvectors, mean_vector):

    return X_reduced @ eigenvectors.T + mean_vector

# wczytanie zbioru digits
digits = datasets.load_digits()
X = digits.data
N, d = X.shape  # N = 1797, d = 64

rmse_values = []

# obliczamy PCA raz w pełnym wymiarze
X_full, eigvecs, eigvals, mean_vec = wiPCA(X, n_components=d)

# kolejne liczby składowych
for k in range(1, d + 1):
    # redukcja do k wymiarów
    X_red_k = X_full[:, :k]
    eigvecs_k = eigvecs[:, :k]

    # rekonstrukcja
    X_rec = wiPCA_inverse(X_red_k, eigvecs_k, mean_vec)

    # RMSE
    rmse = np.sqrt(np.mean(np.sum((X - X_rec) ** 2, axis=1)))
    rmse_values.append(rmse)

# wykres RMSE
plt.figure(figsize=(8, 6))
plt.plot(range(1, d + 1), rmse_values, marker='o')
plt.xlabel('Liczba składowych głównych')
plt.ylabel('RMSE rekonstrukcji')
plt.title('Zadanie 9: Błąd rekonstrukcji PCA (digits)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Zadanie nr 10

# wczytanie zbioru digits
digits = datasets.load_digits()
X = digits.data
y = digits.target

# wybór kilku przykładowych obrazów (np. 5 pierwszych)
indices = [0, 1, 2, 3, 4]
X_samples = X[indices]

# liczby składowych głównych
k_values = [2, 4, 10, 50]

# PCA na pełnym zbiorze
X_full, eigvecs, eigvals, mean_vec = wiPCA(X, n_components=X.shape[1])

# przygotowanie figure
fig, axes = plt.subplots(len(indices), len(k_values) + 1, figsize=(12, 8))

for i, idx in enumerate(indices):
    # oryginalny obraz
    axes[i, 0].imshow(X[idx].reshape(8, 8), cmap='gray')
    axes[i, 0].set_title(f'Oryginał\nCyfra {y[idx]}')
    axes[i, 0].axis('off')

    for j, k in enumerate(k_values):
        # redukcja i rekonstrukcja pojedynczego obrazu
        X_red = X_full[idx, :k].reshape(1, -1)
        eig_k = eigvecs[:, :k]
        X_rec = wiPCA_inverse(X_red, eig_k, mean_vec)

        axes[i, j + 1].imshow(X_rec.reshape(8, 8), cmap='gray')
        axes[i, j + 1].set_title(f'k = {k}')
        axes[i, j + 1].axis('off')

plt.suptitle('Zadanie 10: Rekonstrukcja cyfr przy użyciu PCA', fontsize=14)
plt.tight_layout()
plt.show()