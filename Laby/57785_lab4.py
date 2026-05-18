import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image
import cv2

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

img = np.array(Image.open('image.png'))
print(f"Liczba wymiarów macierzy obrazu: {img.ndim}")
print(f"Kształt macierzy: {img.shape[0]}x{img.shape[1]}")
print(f"Liczba wartości opisujących pojedynczy piksel: {img[img.shape[0]//2, img.shape[1]//2].size}")
plt.figure("Zadanie nr 4: Oryginalny obraz")
plt.imshow(img)
plt.figure("Zadanie nr 4: Resized 98%")
plt.imshow(np.resize(img, (int(img.shape[0]*0.98), int(img.shape[1]*0.98), img.shape[2])))

# Zadanie nr 5

print("Zadanie nr 5")

def convert_1st(Image):
    R = Image[:, :, 0].astype(float)
    G = Image[:, :, 1].astype(float)
    B = Image[:, :, 2].astype(float)
    return ((R + G + B) / 3).astype(np.uint8)

def convert_2nd(Image):
    R = Image[:, :, 0].astype(float)
    G = Image[:, :, 1].astype(float)
    B = Image[:, :, 2].astype(float)
    maxv = np.maximum(np.maximum(R, G), B)
    minv = np.minimum(np.minimum(R, G), B)
    return ((maxv + minv) / 2).astype(np.uint8)

def convert_3rd(Image):
    R = Image[:, :, 0].astype(float)
    G = Image[:, :, 1].astype(float)
    B = Image[:, :, 2].astype(float)
    return (0.21*R + 0.72*G + 0.07*B).astype(np.uint8)

img_method_1 = convert_1st(img.copy())
img_method_2 = convert_2nd(img.copy()) 
img_method_3 = convert_3rd(img.copy())
figure, ax = plt.subplots(1, 3, num="Zadanie nr 5")
figure.suptitle("Zadanie nr 5: Konwersja do skali szarości")
ax[0].imshow(img_method_1, cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Metoda 1")
ax[1].imshow(img_method_2, cmap='gray', vmin=0, vmax=255)
ax[1].set_title("Metoda 2")
ax[2].imshow(img_method_3, cmap='gray', vmin=0, vmax=255)
ax[2].set_title("Metoda 3")
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')

# Zadanie nr 6

print("Zadanie nr 6")

figure, ax = plt.subplots(3, 2, num="Zadanie nr 6")
figure.suptitle("Zadanie nr 6: Histogramy")
ax[0, 0].hist(img_method_1.flatten(), bins=256, range=(0, 255), color='gray')
ax[0, 0].set_title("Metoda 1")
ax[1, 0].hist(img_method_2.flatten(), bins=256, range=(0, 255), color='gray')
ax[1, 0].set_title("Metoda 2")
ax[2, 0].hist(img_method_3.flatten(), bins=256, range=(0, 255), color='gray')
ax[2, 0].set_title("Metoda 3")
ax[0, 1].hist(img_method_1.flatten(), bins=16, range=(0, 255), color='gray')
ax[0, 1].set_title("Metoda 1 - 16 binów")
ax[1, 1].hist(img_method_2.flatten(), bins=16, range=(0, 255), color='gray')
ax[1, 1].set_title("Metoda 2 - 16 binów")
ax[2, 1].hist(img_method_3.flatten(), bins=16, range=(0, 255), color='gray')
ax[2, 1].set_title("Metoda 3 - 16 binów")
ax[0, 0].set_xlim(0, 255)
ax[1, 0].set_xlim(0, 255)
ax[2, 0].set_xlim(0, 255)
ax[0, 1].set_xlim(0, 255)
ax[1, 1].set_xlim(0, 255)
ax[2, 1].set_xlim(0, 255)
plt.tight_layout()
counts, edges = np.histogram(img_method_3.flatten(), bins=16, range=(0, 255))
print("Przedzialy:", edges)


# Zadanie nr 7

print("Zadanie nr 7")

srodki = (edges[:-1] + edges[1:]) / 2
indeksy = np.digitize(img_method_3.flatten(), edges) - 1
indeksy = np.clip(indeksy, 0, len(srodki)-1)
img_q = srodki[indeksy].reshape(img_method_3.shape)

figure, ax = plt.subplots(1, 2, num="Zadanie nr 7")
figure.suptitle("Zadanie nr 7: Kwantyzacja")
ax[0].imshow(img_method_3, cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Oryginalny obraz (luminacja)")
ax[1].imshow(img_q, cmap='gray', vmin=0, vmax=255)
ax[1].set_title("Obraz po kwantyzacji")
ax[0].axis('off')
ax[1].axis('off')

# Zadanie nr 8

print("Zadanie nr 8")

gradient_gray = np.array(Image.open('gradient_obiekt.png'))


figure, ax = plt.subplots(1, 2, num="Zadanie nr 8")
figure.suptitle("Zadanie nr 8: Gradient")
ax[0].imshow(gradient_gray, cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Obraz w skali szarości")
ax[1].hist(gradient_gray.flatten(), bins=256, range=(0, 255), color='gray')
ax[1].set_title("Histogram obrazu")
ax[0].axis('off')
ax[1].set_xlim(0, 255)
plt.tight_layout()

# Zadanie nr 9

print("Zadanie nr 9")

prog_reczny = 60

obraz_uint8 = gradient_gray.astype(np.uint8)
prog_otsu, _ = cv2.threshold(obraz_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Prog Otsu: {prog_otsu}")

plt.figure(num="Zadanie nr 9")
plt.title("Progi binaryzacji")
plt.hist(gradient_gray.flatten(), bins=256, range=(0, 255), color='gray')
plt.axvline(x=prog_reczny, color='red', linestyle='-', label="Próg wyznaczony ręcznie")
plt.axvline(x=prog_otsu, color='blue', linestyle='-', label="Próg wyznaczony metodą Otsu")
plt.xlabel("Wartość piksela")
plt.ylabel("Częstotliwość")
plt.legend()

# Zadanie nr 10

print("Zadanie nr 10")
bin_reczny = (gradient_gray<prog_reczny).astype(np.uint8)
bin_otsu = (gradient_gray<prog_otsu).astype(np.uint8)


fig, ax = plt.subplots(1, 3, num="Zadanie nr 10")
fig.suptitle("Zadanie nr 10")
ax[0].imshow(gradient_gray, cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Oryginalny obraz w skali szarości")
ax[1].imshow(bin_reczny * 255, cmap='gray', vmin=0, vmax=255)
ax[1].set_title("Wynik binaryzacji z progiem ręcznym")
ax[2].imshow(bin_otsu * 255, cmap='gray', vmin=0, vmax=255)
ax[2].set_title("Wynik binaryzacji metodą Otsu")
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
plt.show()