import pandas as pd
import numpy as np

# Zadania podstawowe
# Zadanie nr 1

print("Zadanie nr 1")
arr1 = {
    "A": np.random.randn(5),
    "B": np.random.randn(5),
    "C": np.random.randn(5)
}
idx1 = pd.Index(["2020-03-01","2020-03-02","2020-03-03","2020-03-04","2020-03-05"], name="data")
df1 = pd.DataFrame(arr1, index = idx1)
print(df1) # tabela
print(df1.dtypes) # typy danych w kolumnach

# Zadanie nr 2

print("Zadanie nr 2")
arr2 = {
    'A': np.random.randint(-100, 101, 20),
    'B': np.random.randint(-100, 101, 20),
    'C': np.random.randint(-100, 101, 20)
}
idx2 = pd.Index(range(0,20, 1), name="id")
df2 = pd.DataFrame(arr2, index = idx2)
print(df2)
print(df2.loc[[0, 1, 2]]) # trzy pierwsze wiersze
print(df2.loc[[17, 18, 19]]) # trzy ostatnie wiersze
print(df2.index.name) # nazwa indeksu tabeli
print(df2.columns.tolist()) # nazwy kolumn
print(df2.values) # same wartości
print(df2.sample(5)) # 5 losowych wierszy

# Zadanie nr 3

print("Zadanie nr 3")
print(df2['A']) # kolumna A
print(df2[['A','B']]) # kolumny A i B
print(df2[['A','B']].to_numpy) # kolumny A i B jako tablica numpy

# Zadanie nr 4

print("Zadanie nr 4")
print(df2.iloc[[0, 1, 2], [0, 1]]) # trzy pierwsze wiersze oraz dwie pierwsze kolumny
print(df2.iloc[[5]]) # wiersz o indeksie 5
print(df2.iloc[[0, 5, 6, 7], [1, 2]]) # wiersze o indeksach 0, 5, 6, 7 oraz kolumny o indeksach 1 i 2

# Zadanie nr 5

print("Zadanie nr 5")
print(df2.describe())
print(df2[df2>0])
df2[df2>0][df2]
print(df2[df2['A']>0]['A'])

# Zadanie nr 6

print("Zadanie nr 6")
print(df2.mean()) # średnia dla kolumn
print(df2.mean(axis=1)) # średnia dla wierszy

# Zadanie nr 7

print("Zadanie nr 7")
df71 = pd.DataFrame({
    'A': np.random.randint(-10, 10, 5),
    'B': np.random.randint(-10, 10, 5),
    'C': np.random.randint(-10, 10, 5)
}, index = pd.Index([0, 1 ,2, 3, 4], name = "id"))
df72 = pd.DataFrame({
    'A': np.random.randint(-10, 10, 5),
    'B': np.random.randint(-10, 10, 5),
    'C': np.random.randint(-10, 10, 5)
}, index = pd.Index([0, 1 ,2, 3, 4], name = "id"))
df7 = pd.concat([df71, df72])
print(df7)
print(df7.transpose())

# Zadanie nr 8

print("Zadanie nr 8")
print(df7.sort_index(ascending=False)) # posortowano malejąco według indeksu
print(df7.sort_values(by=['B'], ascending=True)) # posortowano rosnąco według kolumny B

# Zadanie nr 9

print("Zadanie nr 9")
df9 = pd.DataFrame({
    'Day': ['Mon', 'Mon', 'Tue', 'Tue'],
    'Fruit': ['Apple', 'Banana', 'Apple', 'Banana'],
    'Sales': [10, 12, 9 ,7]
})
print(df9.groupby(['Day']).sum()) # pogrupowano według dni i obliczono sumę sprzedaży
print(df9.groupby(['Day', 'Fruit']).sum()) # pogrupowano według dni i owoców i obliczono sumę dla każdej kombinacji

# Zadanie nr 10

print("Zadanie nr 10")
print(df7)
df7['B'] = 1 # Każda wartość w kolumnie B przyjmuje wartość 1
print(df7)
df7.iloc[1, 2] = 10 # Pole o indeksie [1, 2] w tabeli przyjmuje wartość 10
print(df7)
df7[df7<0] = -df7 # Każda wartość mniejsza od 0 zostaje zamieniona na liczbę przeciwną
print(df7)