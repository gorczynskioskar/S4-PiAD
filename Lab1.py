import numpy as np

print("Tablice.")
a = np.array([1,2,3,4,5,6,7]) 
b = np.array([[1,2,3,4,5], [6,7,8,9,10]])
b=np.transpose(b)

a = np.arange(1, 101, 1)
print(a)

a = np.linspace(0, 2, 10)
print(a)

a = np.arange(0, 101, 5)

print("Liczby losowe.")
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

print("Selekcja danych.")
b = np.array([[1,2,3,4,5], [6,7,8,9,10]], dtype=np.int32)
print(np.ndim(b)) #2
print(np.size(b)) #10
print(np.where(b==2)) #b[0][1]
print(np.where(b==4)) #b[0][3]
print(b[0])
print(b[:,0])
b = np.random.randint(0,101, [20, 7])
print(b[:, range(0,4)])

print("Działania matematyczne i logiczne.")
a = np.random.randint(1,11,[3,3])
print(f"a:\n {a}")
b = np.random.randint(1,11,[3,3])
print(f"b:\n {b}")

print("Dodawanie:")
print(a+b)
print(np.add(a,b))

print("Odejmowanie:")
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

print("Dane statystyczne.")
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
print("numpy.ravel służy do \"spłaszczania\" macierzy.")

a = np.reshape(np.arange(100, 200, 5), [5,4])
b = np.reshape(np.arange(50, 100, 2.5), [5,4])


