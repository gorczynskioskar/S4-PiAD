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
print(df2.describe()) # statystyki opisowe
print(df2>0) # sprawdzenie, które wartości są większe od 0
df2_new=pd.concat([df2[df2>0]['A'], df2[df2>0]['B']]) # połączenie wartości większych od 0 kolumn A i B (typ Series)
df2_new=pd.concat([df2_new, df2[df2>0]['C']]) # dodanie wartości większych od 0 kolumny C (typ Series)
df2_new.dropna(inplace=True) # odrzucenie wartości NaN
df2_new = pd.DataFrame({
    'A': df2_new.to_numpy() # stworzenie DataFrame z typu Series przekonwertowanego na ndarray
})

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

# Zadania podsumowujące
# Zadanie nr 11

print("Zadanie nr 11")
df11 = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100),
    'D': np.random.randn(100)
})
df11_warunek = df11[(df11['A']>0) & (df11['B']<0)] # tylko wiersze gdzie kolumna A>0 oraz B<0
print(df11_warunek)
df11_warunek_mean = df11_warunek.mean(axis=0)
print(df11_warunek_mean) # średnia dla każdej kolumny

# Zadanie nr 12

print("Zadanie nr 12")
products = pd.DataFrame({
    "ProductID": [0, 1, 2, 3, 4, 5],
    "Name": ["Tomato", "Rice", "Bread", "Onion", "Cucumber", "Carrot"]
})
sales = pd.DataFrame({
    "ProductID": [0, 1, 2, 3, 6, 7],
    "Sales": np.random.randint(10, 101, 6)
})

merged = pd.merge(products, sales, how = "left") # pobiera wspólny klucz z lewej tablicy, zasada działania jak LEFT JOIN w SQLu
print(merged)
merged = pd.merge(products, sales, how = "inner") # pobiera wspólny klucz z obu tablic, zasada działania jak INNER JOIN w SQLu
print(merged)
merged = pd.merge(products, sales, how = "right") # pobiera wspólny klucz z prawej tablicy, zasada działania jak RIGHT JOIN w SQLu
print(merged)

# Zadanie nr 13

print("Zadanie nr 13")
df13 = pd.DataFrame({
    "Day": ["1-03-2026", "2-03-2026", "3-03-2026", "1-03-2026", "2-03-2026", "3-03-2026", "1-03-2026", "2-03-2026", "3-03-2026"],
    "Product": ["Tomato", "Tomato", "Tomato", "Yoghurt", "Yoghurt", "Yoghurt", "Rice", "Rice", "Rice"],
    "Sales": np.random.randint(10, 101, 9)
})
print(df13)
pivot13 = pd.pivot_table(df13, values="Sales", index=["Day"], aggfunc="sum") # suma sprzedaży dla każdego produktu w poszczególnych dniach
pivot13 = pd.pivot_table(df13, values="Sales", index=["Day"], aggfunc="mean") # średnia sprzedaży jednego produktu w poszczególnych dniach

# Zadanie nr 14

print("Zadanie nr 14")
df14 = pd.DataFrame({
    'A': np.random.randint(10, 101, 10),
    'B': np.random.randint(10, 101, 10),
    'C': np.random.randint(10, 101, 10),
    'D': np.random.randint(10, 101, 10)
    }, index = pd.Index(range(0, 10, 1), name="id"))

df14.index.name = "idx" # zmiany nazwy indeksu
df14.set_index('A', inplace=True) # ustawienie kolumny A jako indeks
df14.reset_index(inplace=True) # zresetowanie indeksu

# Zadanie nr 15

print("Zadanie nr 15")
df14["A+B"] = df14['A']+df14['B'] # suma kolumn A i B
df14["Średnia(A,B,C)"] = df14[['A','B','C']].mean(axis=1) # średnia kolumn A, B i C

df14["Norm A"] = (df14["A"] - df14["A"].mean()) / df14["A"].std(ddof=0) # wartości znormalizowane dla kolumny A
df14["Norm B"] = (df14["B"] - df14["B"].mean()) / df14["B"].std(ddof=0) # wartości znormalizowane dla kolumny B
df14["Norm C"] = (df14["C"] - df14["C"].mean()) / df14["C"].std(ddof=0) # wartości znormalizowane dla kolumny C