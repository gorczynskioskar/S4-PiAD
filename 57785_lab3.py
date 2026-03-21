import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp

# Zadania podstawowe

# Zadanie nr 1

print("Zadanie nr 1")
data1 = {
    'X': [10, 15, 10, 20, 15, 30],
    'Y': ['A', 'A', 'B', 'B', 'A', 'B']
}

df1 = pd.DataFrame(data1)
print(df1.groupby(by='Y').mean())

# Zadanie nr 2

print("Zadanie nr 2")
print(df1.value_counts(subset='Y', ascending=False)) # bez zmian dla sortowania malejącego (liczności są równe)

# Zadanie nr 3

print("Zadanie nr 3")
np3 = np.loadtxt('autos.csv', delimiter=',', dtype = np.str_, skiprows=1)
print(np3[0:5, :])
df3 = pd.read_csv('autos.csv', delimiter=',')
print(df3.head(5))

# Obsługa plików poprzez pandas.DataFrame jest o wiele bardziej przystępna.
# Ułożenie kolumn jest czytelniejsze i dodatkowo jest tu obsługa nagłówków.
# Uważam, że numpy.loadtxt nie jest najzwyczajniej dobrze dostosowane do obsługi plików tego typu.

# Zadanie nr 4

print("Zadanie nr 4")
df4 = df3
print(df4.head(5))
print(df4.columns)
print(df4.dtypes)
print(df4.describe())

# Zadanie nr 5

print("Zadanie nr 5")
df5 = df4.groupby(by="make").mean("city-mpg").sort_values("city-mpg",ascending=False)["city-mpg"]
print(df5)

# Zadanie nr 6

print("Zadanie nr 6")
df6 = df4.groupby(by="make")["fuel-type"].value_counts().unstack(fill_value=0)
print(df6)

# Zadanie nr 7

print("Zadanie nr 7")

df7 = df4[['length', 'city-mpg']].dropna()
length = df4['length'].to_numpy()
citympg = df4['city-mpg'].to_numpy()
np7fit = np.polyfit(length, citympg, 1)
np7val = np.polyval(np7fit, length)
print(f"Współczynniki (a, b): {np7fit}")

# Zadanie nr 8

print("Zadanie nr 8")
np8fit = np.polyfit(length, citympg, 2)
np8val = np.polyval(np8fit, length)
print(f"Współczynniki (a, b, c): {np8fit}")

# Model 2. stopnia lepiej oddaje zależność między zmiennymi, gdyż może do nich lepiej się dostosować jeśli są tam jakieś nieliniowe zależności.

# Zadanie nr 9

print("Zadanie nr 9")
sp9 = sp.pearsonr(length, citympg)
print(f"Korelacja między 'length' a 'city-mpg': {sp9.statistic}")

# Współczynnik korelacji wynosi około -0.67. Oznacza to dość silną zależność malejącą.
# Dla naszych danych oznacza to, że wraz ze wzrostem długości samochodu spada zużycie paliwa w mieście.

# Zadanie nr 10

print("Zadanie nr 10")
sorted_idx = np.argsort(length)
lengths = length[sorted_idx]
model1 = np7val[sorted_idx]
model2 = np8val[sorted_idx]
plt.figure("Zadanie nr 10")
plt.scatter(length, citympg, label="Dane rzeczywiste", alpha=0.6)
plt.plot(lengths, model1, color='red', label="Model 1. stopnia (liniowy)")
plt.plot(lengths, model2, color='green', label="Model 2. stopnia (kwadratowy)")
plt.xlabel("Długość")
plt.ylabel("Mpg w mieście")
plt.legend()
plt.grid(True)

# Zadania podsumowujące

# Zadanie nr 11

print("Zadanie nr 11")
kde11 = sp.gaussian_kde(length)
lin_length = np.linspace(length.min(), length.max(), 500)
density11 = kde11(lin_length)

plt.figure("Zadanie nr 11")
plt.hist(length, bins=30, density=True, alpha=0.5, label="Histogram długości (length)")
plt.plot(lin_length, density11, color='red', label="Estymacja gęstości (KDE)")

plt.title("Rozkład zmiennej length z estymacją gęstości KDE")
plt.xlabel("Długość")
plt.ylabel("Gęstość")
plt.legend()
plt.grid(True)

# Zadanie nr 12

print("Zadanie nr 12")
width = df4["width"].dropna().to_numpy()
fig, (ax1, ax2) = plt.subplots(1, 2, num="Zadanie nr 12")
fig.suptitle("Rozkłady zmiennych length i width")
ax1.hist(length, bins=30, density=True, alpha=0.5, label="Histogram długości (length)")
ax1.set_xlabel("Length (długość)")
ax1.set_ylabel("Rozkład")
ax2.hist(width, bins=30, density=True, alpha=0.5, label="Histogram szerokości (width)")
ax2.set_xlabel("Width (szerokość)")

# Zadanie nr 13

print("Zadanie nr 13")
df13 = df4[['width', 'length']].dropna()
x = df13['width'].to_numpy()
y = df13['length'].to_numpy()
kde13 = sp.gaussian_kde([x, y])
xi, yi = np.meshgrid(
    np.linspace(x.min(), x.max(), 200),
    np.linspace(y.min(), y.max(), 200)
)

xy = np.vstack([xi.ravel(), yi.ravel()])
zi = kde13(xy).reshape(xi.shape)

plt.figure("Zadanie nr 13")
plt.scatter(x, y, s=10, alpha=0.4, label="Dane (width/length)")
plt.contour(xi, yi, zi, levels=15, cmap="viridis")
plt.title("Dwuwymiarowa estymacja gęstości KDE dla width i length")
plt.xlabel("Width")
plt.ylabel("Length")
plt.legend()
plt.grid(True)


# Zadanie nr 14

print("Zadanie nr 14")
plt.savefig("zadanie_13.png", dpi=300)
plt.savefig("zadanie_13.pdf", dpi=300)
if(os.path.exists("zadanie_13.png") and os.path.exists("zadanie_13.pdf")):
    print("Pliki zostały zapisane poprawnie.")
plt.show()

# Zadanie nr 15

print("Zadanie nr 15")
pandas_len_mean = df4["length"].dropna().mean()
numpy_len_mean = np.mean(df4["length"].dropna())

pandas_width_std = df4["width"].dropna().std()
numpy_width_std = np.std(df4["width"].dropna(), ddof=1) # ddof=1 dla odchylenia standardowego, aby uzyskać zgodność z pandas

print(f"Średnia długość (pandas): {pandas_len_mean}")
print(f"Średnia długość (numpy): {numpy_len_mean}")
print(f"Odchylenie standardowe szerokości (pandas): {pandas_width_std}")
print(f"Odchylenie standardowe szerokości (numpy): {numpy_width_std}")

# Wyniki są identyczne, co potwierdza, że zarówno pandas, jak i numpy poprawnie obliczają średnią i odchylenie standardowe dla tych danych.
# Wymagało to jednak jawnego ustawienia ddof=1 w numpy, aby uzyskać zgodność z pandas, który domyślnie używa tej wartości dla odchylenia standardowego (w numpy jest to domyślnie 0).