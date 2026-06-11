# Zadanie nr 1

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generowanie danych
X, y = datasets.make_classification(
    n_samples=2000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=4,
    n_clusters_per_class=1,
    random_state=42
)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42
)

# Klasyfikacja binarna - 2 klasy
# Klasyfikacja wieloklasowa - więcej niż 2 klasy

# OvR - każdy klasyfikator rozróżnia jedną klasę vs reszta
# OvO - klasyfikatory dla każdej pary klas

# Zadanie nr 2

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression, Perceptron

# Klasyfikatory
classifiers = {
    "OvO SVC linear": OneVsOneClassifier(svm.SVC(kernel='linear', probability=True)),
    "OvO SVC rbf": OneVsOneClassifier(svm.SVC(kernel='rbf', probability=True)),
    "OvO LogisticRegression": OneVsOneClassifier(LogisticRegression(max_iter=1000)),
    "OvO Perceptron": OneVsOneClassifier(Perceptron()),

    "OvR SVC linear": OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)),
    "OvR SVC rbf": OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True)),
    "OvR LogisticRegression": OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    "OvR Perceptron": OneVsRestClassifier(Perceptron())
}

results_predictions = {}
results_scores = {}

for name, clf in classifiers.items():

    clf.fit(X_train, y_train)

    # Predykcja klas
    y_pred = clf.predict(X_test)
    results_predictions[name] = y_pred

    if hasattr(clf, "predict_proba"):
        try:
            y_score = clf.predict_proba(X_test)
        except:
            y_score = None
    else:
        y_score = None

    if y_score is None and hasattr(clf, "decision_function"):
        try:
            y_score = clf.decision_function(X_test)
        except:
            y_score = None

    # zapis wyników
    results_scores[name] = y_score

# Zadanie nr 3

from sklearn import metrics
import pandas as pd

results_metrics = []

for name in classifiers.keys():

    y_pred = results_predictions[name]
    y_score = results_scores[name]

    # Metryki klasyczne
    accuracy = metrics.accuracy_score(y_test, y_pred)

    recall = metrics.recall_score(
        y_test, y_pred, average="macro"
    )

    precision = metrics.precision_score(
        y_test, y_pred, average="macro"
    )

    f1 = metrics.f1_score(
        y_test, y_pred, average="macro"
    )

    # AUC
    if y_score is not None:
        try:
            auc = metrics.roc_auc_score(
                y_test,
                y_score,
                multi_class="ovr"
            )
        except Exception:
            auc = None
    else:
        auc = None

    # Zapis wyników
    results_metrics.append({
        "classifier": name,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auc": auc
    })

# DataFrame
results_df = pd.DataFrame(results_metrics)
results_df = results_df.set_index("classifier")

print(results_df)

# Zadanie nr 4

for name in classifiers.keys():

    y_pred = results_predictions[name]

    fig, axs = plt.subplots(1, 3, figsize=(6, 4))

    # oczekiwane klasy
    axs[0].scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="tab10",
        alpha=0.5
    )
    axs[0].set_title("oczekiwane")

    # predykcje
    axs[1].scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_pred,
        cmap="tab10",
        alpha=0.5
    )
    axs[1].set_title("obliczone")

    # Różnice

    correct = (y_pred == y_test)
    incorrect = (y_pred != y_test)

    # poprawne
    axs[2].scatter(
        X_test[correct, 0],
        X_test[correct, 1],
        color="green",
        label="Poprawne"
    )

    # błędne
    axs[2].scatter(
        X_test[incorrect, 0],
        X_test[incorrect, 1],
        color="red",
        label="Błędne"
    )

    axs[2].set_title("różnice")

    for ax in axs:
        ax.set_xlabel("Cecha 1")
        ax.set_ylabel("Cecha 2")
        ax.grid(True)

    fig.suptitle(f"Wyniki klasyfikacji - {name}")

    plt.tight_layout()
    plt.show()

# Zadanie nr 5

# Metryki
metrics_to_plot = results_df[
    ["accuracy", "recall", "precision", "f1", "auc"]
]

metrics_to_plot.plot(kind="bar")

plt.title("Porównanie jakości klasyfikatorów (OvO vs OvR)")
plt.xlabel("Klasyfikator")
plt.ylabel("Wartość metryki")

plt.xticks(rotation=45)
plt.legend(title="Metryka")

plt.grid(axis="y")

plt.tight_layout()
plt.show()

# Zadanie nr 6

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Klasy
classes = np.unique(y_test)

# Binarizacja etykiet
y_test_bin = label_binarize(y_test, classes=classes)

# Pętla po klasyfikatorach
for name in classifiers.keys():

    y_score = results_scores[name]

    if y_score is None:
        print(f"{name} - brak predict_proba (pominięto ROC)")
        continue

    plt.figure(num=f"Zadanie nr 6: {name}", figsize=(7, 6))

    # ROC dla każdej klasy
    for i in range(len(classes)):

        # Obliczenie FPR i TPR
        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )

        # Obliczenie AUC
        roc_auc = auc(fpr, tpr)

        # Rysowanie krzywej
        plt.plot(
            fpr,
            tpr,
            label=f"Klasa {i} (AUC = {roc_auc:.3f})"
        )

    # Linia odniesienia
    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Linia odniesienia"
    )

    # Opisy wykresu
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Krzywe ROC - {name}")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.show()

# Zadanie nr 7

import numpy as np
import matplotlib.pyplot as plt

# Zakres przestrzeni cech
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Tworzenie siatki punktów
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Spłaszczenie siatki
grid = np.c_[xx.ravel(), yy.ravel()]

# Pętla po wszystkich klasyfikatorach
for name, clf in classifiers.items():

    # Trening modelu
    clf.fit(X_train, y_train)

    # Predykcja dla siatki
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    # Tworzenie wykresu
    plt.figure(num=f"Zadanie nr 7: {name}", figsize=(6, 5))

    # Tło
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")

    # Punkty testowe
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="tab10",
        edgecolor="k"
    )

    # Opisy
    plt.title(f"Granica decyzyjna - {name}")
    plt.xlabel("Cecha 1")
    plt.ylabel("Cecha 2")
    plt.grid(True)

    plt.show()

# Zadanie nr 8

# Najlepsze wyniki zazwyczaj osiąga klasyfikator SVC z jądrem RBF,
# ponieważ modeluje nieliniowe zależności między klasami.

# Strategia OvO często daje lepsze wyniki niż OvR,
# ponieważ buduje osobne klasyfikatory dla każdej pary klas,
# co pozwala lepiej uchwycić różnice między klasami.

# Strategia OvR jest prostsza i szybsza obliczeniowo,
# ale może gorzej radzić sobie z bardziej złożonymi danymi.

# Nie istnieje jeden klasyfikator najlepszy dla wszystkich metryk,
# ponieważ każda metryka mierzy inny aspekt jakości klasyfikacji.

# Modele liniowe (SVC linear, LogisticRegression, Perceptron)
# tworzą proste granice decyzyjne.

# Model SVC z jądrem RBF tworzy bardziej złożone, nieliniowe granice,
# które lepiej dopasowują się do danych.

# Krzywe ROC pokazują zdolność modelu do rozróżniania klas.
# Wyższe wartości AUC oznaczają lepszą jakość klasyfikacji.

# W klasyfikacji wieloklasowej ROC liczony jest osobno dla każdej klasy
# w podejściu one-vs-rest.

# Metryka AUC nie zawsze jest dostępna dla wszystkich klasyfikatorów,
# ponieważ wymaga danych w postaci (n_samples, n_classes).
# W przypadku OvO oraz Perceptronu nie zawsze jest to spełnione.