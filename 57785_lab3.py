import numpy as np
import pandas as pd
import matplotlib as plt
import scipy as sp

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
df6 = df4.groupby(by="make").value_counts(["fuel-type"])
#df6 = pd.pivot_table(df6, aggfunc='count')
print(df6)

# Zadanie nr 7

print("Zadanie nr 7")
length = df4['length'].to_numpy()
citympg = df4['city-mpg'].to_numpy()
np7fit = np.polyfit(length, citympg, 1)
np7val = np.polyval(length, citympg)
print(np7fit)

# Zadanie nr 8

print("Zadanie nr 8")
np8fit = np.polyfit(length, citympg, 2)
np8val = np.polyval(length, citympg)
print(np8fit)

# Model 2. stopnia lepiej oddaje zależność między zmiennymi, gdyż może do nich lepiej się dostosować.

# Zadanie nr 9

print("Zadanie nr 9")
