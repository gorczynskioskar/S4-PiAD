from sklearn import datasets
import matplotlib.pyplot as plt

# Zadanie nr 1

X, y = datasets.make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=2,
    n_classes=2,
    random_state=42
)

plt.figure(num="Zadanie nr 1", figsize=(6, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1],
            color='blue', label='Klasa 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            color='red', label='Klasa 1')

plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.title('Wygenerowane dane binarne')
plt.legend()
plt.grid(True)

plt.show()

import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Zadanie nr 2

# Lista klasyfikatorów
classifiers = {
    "GaussianNB": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "kNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

results = []
last_iteration_results = {}

# Pętla po klasyfikatorach
for name, clf in classifiers.items():
  # 100 iteracji
    for i in range(100):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i
        )

        # Czas uczenia
        start_train = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_train

        # Czas testowania
        start_test = time.time()
        y_pred = clf.predict(X_test)
        test_time = time.time() - start_test

        # Prawdopodieństwa dla klasy pozytywnej (do AUC)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Metryki jakości klasyfikacji
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_proba)

        # Zapis wyników
        results.append({
            "classifier": name,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "auc": auc,
            "train_time": train_time,
            "test_time": test_time
        })

        # Zapis ostatniej iteracji
        if i == 99:
          last_iteration_results[name] = {
              "model": clf,
              "X_train": X_train,
              "y_train": y_train,
              "X_test": X_test,
              "y_test": y_test,
              "y_proba": y_proba,
              "y_pred": y_pred
          }

results_df = pd.DataFrame(results)

print(results_df.head())

# accuracy  - odsetek poprawnych klasyfikacji
# recall    - czułość, jaki procent klasy pozytywnej wykryto
# precision - ile predykcji pozytywnych było poprawnych
# f1        - kompromis między precision i recall
# AUC       - zdolność modelu do rozróżniania klas

# Dokładność (accuracy) może być myląca przy niezbalansowanych danych,
# gdy jedna klasa znacząco dominuje liczebnie.

import matplotlib.pyplot as plt

# Zadanie nr 3

# Uśrednienie wyników po 100 iteracjach
mean_results = results_df.groupby("classifier").mean()

# Zaokrąglenie wyników
mean_results = mean_results.round(4)

# Wyświetlenie tabeli wyników
print("Średnie wyniki klasyfikatorów:")
print(mean_results)

# Wybór metryk do wizualizacji
metrics_to_plot = mean_results[
    ["accuracy", "recall", "precision", "f1", "auc"]
]

# Wykres słupkowy
metrics_to_plot.plot(kind="bar")

plt.title("Porównanie jakości klasyfikatorów\n(średnia z 100 iteracji)")
plt.xlabel("Klasyfikator")
plt.ylabel("Wartość miary")
plt.legend(title="Metryka")
plt.grid(axis="y")

plt.tight_layout()
plt.show()

# Najlepszy klasyfikator zależy od wybranej miary jakości.
# Często SVM lub kNN osiągają najwyższe AUC i F1,
# natomiast GaussianNB oraz QDA uczą się najszybciej.
# Nie zawsze ten sam klasyfikator jest najlepszy dla wszystkich miar.

# Zadanie nr 4

import matplotlib.pyplot as plt

# Dane pochodzą z ostatniej iteracji Zadania 2
for name, data in last_iteration_results.items():

    X_test = data["X_test"]
    y_test = data["y_test"]
    y_pred = data["y_pred"]

    # Poprawne i błędne klasyfikacje
    correct = y_pred == y_test
    incorrect = y_pred != y_test

    # Wykres
    plt.figure(num=f"Zadanie nr 4: {name}", figsize=(6, 5))

    plt.scatter(
        X_test[correct, 0],
        X_test[correct, 1],
        color="green",
        label="Poprawnie sklasyfikowane"
    )

    plt.scatter(
        X_test[incorrect, 0],
        X_test[incorrect, 1],
        color="red",
        label="Błędnie sklasyfikowane"
    )

    plt.title(f"Błędy klasyfikacji - {name}")
    plt.xlabel("Cecha 1")
    plt.ylabel("Cecha 2")
    plt.legend()
    plt.grid(True)

    plt.show()

# Zadanie nr 5

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(num="Zadanie nr 5", figsize=(7, 6))

# Krzywe ROC dla wszystkich klasyfikatorów
for name, data in last_iteration_results.items():

    y_test = data["y_test"]
    y_proba = data["y_proba"]

    # Wyznaczenie krzywej ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Rysowanie krzywej
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

# Linia odniesienia – klasyfikator losowy
plt.plot([0, 1], [0, 1], "k--", label="Losowy klasyfikator")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Krzywe ROC dla klasyfikatorów")
plt.legend(loc="lower right")
plt.grid(True)

plt.show()

# Zadanie nr 6

import numpy as np
import matplotlib.pyplot as plt

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Siatka punktów
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Granice decyzyjne dla każdego klasyfikatora
for name, data in last_iteration_results.items():

    model = data["model"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Predykcja na siatce
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Wykres
    plt.figure(num=f"Zadanie nr 6: {name}", figsize=(6, 5))

    plt.contourf(
        xx, yy, Z,
        alpha=0.3,
        cmap="coolwarm"
    )

    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        edgecolor="k"
    )

    plt.title(f"Granica decyzyjna - {name}")
    plt.xlabel("Cecha 1")
    plt.ylabel("Cecha 2")
    plt.grid(True)

    plt.show()


# Drzewo decyzyjne i kNN tworzą najbardziej elastyczne,
# nieregularne granice decyzyjne.
# QDA i SVM generują gładsze granice,
# natomiast Naive Bayes zakłada prostszy rozkład danych.

# Zadanie nr 7

from sklearn.neighbors import KNeighborsClassifier

# Klasyfikator
clf = KNeighborsClassifier()

# Przestrzeń hiperparametrów
param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15],
    "p": [1, 2]
}

# Wybrałem klasyfikator kNN, ponieważ jego działanie silnie zależy
# od liczby sąsiadów oraz metryki odległości.
# Parametr n_neighbors kontroluje stopień wygładzenia granicy decyzyjnej,
# natomiast parametr p określa sposób liczenia odległości między punktami.

# Zadanie nr 8

from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# przygotowanie GridSearchCV
grid = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5
)

grid.fit(X, y)

print("Najlepsze parametry:", grid.best_params_)
print("Najlepsze AUC:", grid.best_score_)

# Wyniki GridSearch
results_df = pd.DataFrame(grid.cv_results_)

# Wykres 1D
plt.figure(num="Zadanie nr 8: Wykres 1D", figsize=(7, 5))

for p in sorted(results_df["param_p"].unique()):
    subset = results_df[results_df["param_p"] == p]
    plt.plot(
        subset["param_n_neighbors"],
        subset["mean_test_score"],
        marker="o",
        label=f"p = {p}"
    )

plt.xlabel("Liczba sąsiadów (n_neighbors)")
plt.ylabel("Średnie AUC (CV)")
plt.title("Wpływ parametrów kNN na jakość klasyfikacji")
plt.legend()
plt.grid(True)
plt.show()

# Wykres 2D
pivot_table = results_df.pivot(
    index="param_p",
    columns="param_n_neighbors",
    values="mean_test_score"
)

plt.figure(num="Zadanie nr 8: Wykres 2D", figsize=(8, 4))
plt.imshow(pivot_table, aspect="auto", cmap="viridis")
plt.colorbar(label="Średnie AUC (CV)")

plt.xticks(
    range(len(pivot_table.columns)),
    pivot_table.columns
)
plt.yticks(
    range(len(pivot_table.index)),
    pivot_table.index
)

plt.xlabel("n_neighbors")
plt.ylabel("p")
plt.title("Mapa ciepła AUC dla parametrów kNN")
plt.show()

# Zadanie nr 9

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import time

# Klasyfikator z optymalnymi parametrami
best_clf = KNeighborsClassifier(**grid.best_params_)

results_opt = []
last_iteration_9 = {}

# 100 iteracji
for i in range(100):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i
    )

    # Czas uczenia
    start_train = time.time()
    best_clf.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Czas testowania
    start_test = time.time()
    y_pred = best_clf.predict(X_test)
    test_time = time.time() - start_test

    # Prawdopodobieństwa
    y_proba = best_clf.predict_proba(X_test)[:, 1]

    # Metryki
    results_opt.append({
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "f1": metrics.f1_score(y_test, y_pred),
        "auc": metrics.roc_auc_score(y_test, y_proba),
        "train_time": train_time,
        "test_time": test_time
    })
    if i == 99:
      last_iteration_9 = {
          "X_train": X_train,
          "y_train": y_train,
          "X_test": X_test,
          "y_test": y_test
      }

# DataFrame i średnie
results_opt_df = pd.DataFrame(results_opt)
mean_opt_results = results_opt_df.mean().round(4)

print("Średnie wyniki klasyfikatora po optymalizacji:")
print(mean_opt_results)

# Porównanie z klasyfikatorem domyślnym
comparison = pd.DataFrame({
    "Domyślny kNN": mean_results.loc["kNN"],
    "kNN po optymalizacji": mean_opt_results
})

print("\nPorównanie wyników:")
print(comparison)

# Optymalizacja hiperparametrów poprawiła jakość klasyfikacji,
# szczególnie pod względem AUC i F1, lecz nie dla każdej metryki.
# Różnice w accuracy są mniejsze, a recall wypada nawet gorzej.
# kNN po optymalizacji przejawia się ponadto wydłużonymi czasami,
# zarówno uczenia jak i testowania.

# Zadanie nr 10

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

# Dane z ostatniej iteracji Zadania 9
X_train = last_iteration_9["X_train"]
y_train = last_iteration_9["y_train"]
X_test = last_iteration_9["X_test"]
y_test = last_iteration_9["y_test"]

# Klasyfikatory
default_clf = KNeighborsClassifier()
optimized_clf = best_clf

# Uczenie
default_clf.fit(X_train, y_train)
optimized_clf.fit(X_train, y_train)

# Krzywe ROC
y_proba_default = default_clf.predict_proba(X_test)[:, 1]
y_proba_opt = optimized_clf.predict_proba(X_test)[:, 1]

fpr_def, tpr_def, _ = roc_curve(y_test, y_proba_default)
fpr_opt, tpr_opt, _ = roc_curve(y_test, y_proba_opt)

auc_def = auc(fpr_def, tpr_def)
auc_opt = auc(fpr_opt, tpr_opt)

plt.figure(num="Zadanie nr 10: krzywe ROC", figsize=(7, 6))
plt.plot(fpr_def, tpr_def, label=f"Domyślny kNN (AUC={auc_def:.3f})")
plt.plot(fpr_opt, tpr_opt, label=f"Optymalny kNN (AUC={auc_opt:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Losowy")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Porównanie krzywych ROC")
plt.legend()
plt.grid(True)
plt.show()

# Granice decyzyjne
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

Z_def = default_clf.predict(grid).reshape(xx.shape)
Z_opt = optimized_clf.predict(grid).reshape(xx.shape)

# Domyślny kNN
plt.figure(num="Zadanie nr 10: Granica decyzyjna - domyślny kNN", figsize=(6, 5))
plt.contourf(xx, yy, Z_def, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolor="k")
plt.title("Granica decyzyjna - domyślny kNN")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.grid(True)
plt.show()

# Optymalny kNN
plt.figure(num="Zadanie nr 10: Granica decyzyjna - optymalny kNN", figsize=(6, 5))
plt.contourf(xx, yy, Z_opt, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolor="k")
plt.title("Granica decyzyjna - kNN po optymalizacji")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.grid(True)
plt.show()