import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image

# Zadanie podstawowe

# Zadanie nr 1

print("Zadanie nr 1")

def generate_signal(f, Fs):
    t_vector = np.arange(0, 1, 1/Fs)
    signal_values = np.sin(2*np.pi*f*t_vector)
    return [t_vector, signal_values]

# Zadanie nr 2

print("Zadanie nr 2")

f=10.0
Fs_vector = [20,21,30,45,50,100,150,200,250,1000]

for n in Fs_vector:
    [t, s] = generate_signal(f, n)
    plt.figure(f"Zadanie nr 2: {n} Hz")
    plt.plot(t, s)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")

# Zadanie nr 3
    
print("Zadanie nr 3")

[t1, s1] = generate_signal(f, 20)
[t2, s2] = generate_signal(f, 1000)
figure, ax = plt.subplots(1, 1, num="Zadanie nr 3")
figure.suptitle("Wykresy 20 Hz vs 1000 Hz")
ax.plot(t1, s1, marker='o', color = 'red', label="20 Hz")
ax.plot(t2, s2, color = 'blue', label="1000 Hz")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Amplituda")
ax.legend()

# Twierdzenie o minimalnej częstotliwości próbkowania mówi, że aby wiernie odtworzyć
# sygnał analogowy z próbek cyfrowy to częstotliwość próbkowania (Fs) musi być co najmniej
# dwukrotnie wyższa niż najwyższa częstotliwość składowa sygnału.

# Zjawisko wynikające z jej niedotrzymania to aliasing.

# Zadanie nr 4

print("Zadanie nr 4")

img = np.array(Image.open('shirt.png'))
print(f"Liczba wymiarów macierzy obrazu: {img.ndim}")
print(f"Kształt macierzy: {img.shape[0]}x{img.shape[1]}")
print(f"Liczba wartości opisujących pojedynczy piksel: {img[1250,1250].size}")
plt.figure("Zadanie nr 4")
plt.imshow(img)
plt.show()

# Zadanie nr 5

print("Zadanie nr 5")

def convert_1st(Image: np.ndarray):
    for row in Image:
        for col in Image:
            R = float(Image[row, col])
            G = float(Image[row, col])
            B = float(Image[row, col])
            mean = (R+G+B)/3
            n[1]=mean

plt.figure("Zadanie nr 5: Metoda nr 1")
plt.imshow(convert_1st(img))
plt.show()
