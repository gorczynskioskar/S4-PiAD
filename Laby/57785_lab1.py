import numpy as np
from numpy.lib.stride_tricks import as_strided

# Tablice
a = np.array([1,2,3,4,5,6,7]) 
b = np.array([[1,2,3,4,5], [6,7,8,9,10]])
b=np.transpose(b)

a = np.arange(1, 101, 1)
print(a)

a = np.linspace(0, 2, 10)
print(a)

a = np.arange(0, 101, 5)

# Liczby losowe
a = np.round(np.random.randn(1, 20), 2)
print(a)

a = np.random.randint(1, 1001, 100)

a = np.zeros([2,3])
a = np.ones([2,3])

a = np.random.randint(0, 101, [5,5], np.int32)

a = np.random.rand(1, 10)*10
print(a)
b = np.array(a, np.int32)
a = np.round(a, 0)
print(a)
print(b)
print("Rzutowanie na typ int jedynie odcina część ułamkową liczby, numpy.round natomiast dba też o zaokrąglanie liczby do góry.")

# Selekcja danych
b = np.array([[1,2,3,4,5], [6,7,8,9,10]], dtype=np.int32)
print(np.ndim(b)) #2
print(np.size(b)) #10
print(np.where(b==2)) #b[0][1]
print(np.where(b==4)) #b[0][3]
print(b[0])
print(b[:,0])
b = np.random.randint(0,101, [20, 7])
print(b[:, range(0,4)])

# Działania matematyczne i logiczne
a = np.random.randint(1,11,[3,3])
print(f"a:\n {a}")
b = np.random.randint(1,11,[3,3])
print(f"b:\n {b}")

# Dodawanie
print(a+b)
print(np.add(a,b))

# Odejmowanie
print(a-b)
print(np.subtract(a,b))

print("Mnożenie:")
print(a*b)
print(np.multiply(a,b))
print(np.dot(a,b))
print(np.matmul(a,b))

print("Dzielenie:")
print(a/b)
print(np.divide(a,b))

print("Potęgowanie:")
print(a**b)
print(np.power(a,b))

print("a>=4?")
print(a>=4)
print("1<=a<=4?")
print(np.logical_and(a<=4, a>=1))
print(f"Suma głównej przekątnej macierzy b = {np.trace(b)}")

# Dane statystyczne
print(f"Suma macierzy b = {np.sum(b)}")
print(f"Minimum macierzy b = {np.min(b)}")
print(f"Maksimum macierzy b = {np.max(b)}")
print(f"Odchylenie standardowe macierzy b = {np.std(b)}")
print("Średnia dla wierszy macierzy b:")
print(np.average(b, axis=1))
print("Średnia dla kolumn macierzy b:")
print(np.average(b, axis=0))

print("Rzutowanie wymiarów za pomocą rehape lub resize:")
a = np.arange(1, 100, 2)
b = np.reshape(a, [10, 5])
b = np.resize(a, [10, 5])
print(b)
b = np.ravel(b)
print(b)
print("numpy.ravel służy do \"spłaszczania\" macierzy.\n")

a = np.reshape(np.arange(15, 40, 5), 5)
b = np.reshape(np.arange(43, 67, 6), 4)
print(f"a: \n{a}\nb:\n{b}")
A = a[:, np.newaxis]
B = b[np.newaxis, :]
print(f"A + B: {A+B}")

# newaxis pozwala rozszerzyć wymiary tablicy, aby umożliwić dodanie tablic o różnych długościach.

print("Sortowanie danych.")
a=np.random.randint(0, 100, [5,5])
print(f"a:\n{a}")
print(f"Wiersze rosnąco:\n{np.sort(a, 1)}")
print(f"Kolumny malejąco:\n{np.flip(np.sort(a, 0))}")

b = np.array([
    [1, 'MZ', 'mazowieckie'],
    [2, 'ZP', 'zachodniopomorskie'],
    [3, 'ML', 'małopolskie']
], dtype=object)

idx = np.argsort(b[:, 1])
b_sorted_by_col2 = b[idx]
print("b posortowane rosnąco po kolumnie 2:")
print(b_sorted_by_col2)

woj_name = b[b[:, 1] == 'ZP', 2][0]
print(f"Nazwa województwa dla skrótu 'ZP': {woj_name}")


# Zadania podsumowujące

# 1.
m1 = np.random.randint(0, 101, (10, 5))
print(f"1. Macierz 10x5:\n{m1}")
print(f"Suma głównej przekątnej (trace): {np.trace(m1)}")
print(f"Wartości na przekątnej (diag): {np.diag(m1)}\n")

# 2.
m1 = np.random.normal(loc=0.0, scale=1.0, size=(5,5))
m2 = np.random.normal(loc=0.0, scale=1.0, size=(5,5))
print(f"2. m1 * m2 :\n{m1 * m2}\n")

# 3.
arrA = np.random.randint(0, 101, 20)
arrB = np.random.randint(0, 101, 20)
A = arrA.reshape(-1, 5)
B = arrB.reshape(-1, 5)
print(f"3. A:\n{A}")
print(f"B:\n{B}")
print(f"A + B:\n{A+B}\n")

# 4.
C = np.random.randint(0, 50, (5,4))
D = np.random.randint(0, 50, (4,5))
print(f"4. C (5x4):\n{C}")
print(f"D (4x5):\n{D}")
sum_C_D = C + np.transpose(D)
print(f"C + D.T:\n{sum_C_D}\n")

# 5.
result = sum_C_D[:, 2] * sum_C_D[:, 3]
print(f"5. Iloczyn kolumn 3 i 4 (z C + D.T):{result}\n")

# 6.
M_norm = np.random.normal(0, 1, (6,6))
M_unif = np.random.uniform(-1, 1, (6,6))
print("\n6. Statystyki dla rozkładu normalnego:")
print(f"Średnia: {M_norm.mean()}")
print(f"Odchylenie std: {M_norm.std()}")
print(f"Wariancja: {M_norm.var()}")
print(f"Suma: {M_norm.sum()}")
print(f"Min: {M_norm.min()}")
print(f"Max: {M_norm.max()}")
print("\nStatystyki dla rozkładu normalnego:")
print(f"Średnia: {M_unif.mean()}")
print(f"Odchylenie std: {M_unif.std()}")
print(f"Wariancja: {M_unif.var()}")
print(f"Suma: {M_unif.sum()}")
print(f"Min: {M_unif.min()}")
print(f"Max: {M_unif.max()}")

# 7.
size = 4
a = np.random.randint(0, 10, (size, size))
b = np.random.randint(0, 10, (size, size))
print(f"\n7. a:\n{a}")
print(f"b:\n{b}")
print(f"a * b (elementowo):\n{a*b}")
print(f"a dot b (iloczyn macierzowy):\n{a.dot(b)}\n")
# * mnoży element po elemencie, natomiast dot wykonuje mnożenie macierzowe. Dlatego warto użyć funkcji dot kiedy chcemy dokonać "prawdziwego" mnożenia macierzy.

# 8.
a = np.arange(100).reshape(10, 10)
print(f"a:\n{a}")
print(f"8. a.strides:", a.strides)
result = as_strided(a, shape=(3,5))
print(f"Widok 3x5 (pierwsze 3 wiersze i 5 kolumn):\n{result}")

# 9.
va = np.array([1,2,3])
vb = np.array([4,5,6])
print(f"\n9. va:\n{va}\nvb:\n{vb}")
print(f"vstack([va, vb]):\n{np.vstack([va, vb])}")
print(f"stack([va, vb], axis=0):\n{np.stack([va, vb], axis=0)}")
print(f"stack([va, vb], axis=1):\n{np.stack([va, vb], axis=1)}")
# vstack łączy macierze w pionie, stack łączy względem nowej osi. stack przydaje się zatem gdy chcemy poszerzyć macierz o dodatkowy wymiar, a vstack gdy chcemy "dokleić" nowe wiersze do macierzy

# 10.
X = np.random.randint(0, 100, (8,8))
height, width = 2, 2
H, W = X.shape
out_shape = (H//height, W//width, height, width)
X_blocks = as_strided(X, shape=out_shape, strides=(X.strides[0]*height, X.strides[1]*width, X.strides[0], X.strides[1])
)
X_block_max = X_blocks.max(axis=(2,3))
print(f"\n10. Macierz X ({H}x{W}):\n{X}")
print(f"Maksimum w blokach {height}x{width}:\n{X_block_max}")