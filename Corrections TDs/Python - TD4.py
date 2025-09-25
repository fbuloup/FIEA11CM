import numpy as np
import matplotlib.pyplot as plt

# Création du vecteur temporel
Fe = 1000  # Fréquence d'échantillonnage
Te = 1 / Fe  # Période d'échantillonnage
nbSamples = 3001  # Nombre d'échantillons
t = np.arange(0, nbSamples) * Te  # Vecteur temporel

# Création des signaux : domaine temporel
s1 = 3 * np.ones(nbSamples)
s2 = 6 * np.sin(2 * np.pi * t)
s3 = 1 + 2 * np.cos(10 * np.pi * t) + 4 * np.cos(20 * np.pi * t)
s4 = np.cos(100 * np.pi * t) + np.cos(100 * np.pi * t) * np.cos(100 * np.pi * t)

# Passage dans le domaine fréquentiel : Série de Fourier
fftS1 = np.fft.fftshift(np.fft.fft(s1) / nbSamples)
fftS2 = np.fft.fftshift(np.fft.fft(s2) / nbSamples)
fftS3 = np.fft.fftshift(np.fft.fft(s3) / nbSamples)
fftS4 = np.fft.fftshift(np.fft.fft(s4) / nbSamples)

# Calcul des modules des fft
ampfftS1 = np.abs(fftS1)
ampfftS2 = np.abs(fftS2)
ampfftS3 = np.abs(fftS3)
ampfftS4 = np.abs(fftS4)

# Calcul des phases des fft
phasefftS1 = np.angle(fftS1)
phasefftS2 = np.angle(fftS2)
phasefftS3 = np.angle(fftS3)
phasefftS4 = np.angle(fftS4)

# Nettoyage des phases
epsilon = 1e-5
phasefftS1[ampfftS1 < epsilon] = 0
phasefftS2[ampfftS2 < epsilon] = 0
phasefftS3[ampfftS3 < epsilon] = 0
phasefftS4[ampfftS4 < epsilon] = 0

# Création du vecteur fréquentiel
df = Fe / nbSamples
if nbSamples % 2 == 1:
    # Nombre d'échantillons impair
    f = np.arange(-Fe/2 + df/2, Fe/2, df)
else:
    # Nombre d'échantillons pair
    f = np.arange(-Fe/2, Fe/2, df)

# Affichage des spectres bilatéraux
plt.figure(figsize=(12, 10))

# Affichage des spectres de s1
plt.subplot(4, 2, 1)
plt.stem(f, ampfftS1)
plt.title("Spectre d'amplitude de s1")
plt.xlabel('f (Hz)')
plt.ylabel('Amplitude')

plt.subplot(4, 2, 2)
plt.stem(f, phasefftS1)
plt.title("Spectre de phase de s1")
plt.xlabel('f (Hz)')
plt.ylabel('Phase (rad)')

# Affichage des spectres de s2
plt.subplot(4, 2, 3)
plt.stem(f, ampfftS2)
plt.title("Spectre d'amplitude de s2")
plt.xlabel('f (Hz)')
plt.ylabel('Amplitude')

plt.subplot(4, 2, 4)
plt.stem(f, phasefftS2)
plt.title("Spectre de phase de s2")
plt.xlabel('f (Hz)')
plt.ylabel('Phase (rad)')

# Affichage des spectres de s3
plt.subplot(4, 2, 5)
plt.stem(f, ampfftS3)
plt.title("Spectre d'amplitude de s3")
plt.xlabel('f (Hz)')
plt.ylabel('Amplitude')

plt.subplot(4, 2, 6)
plt.stem(f, phasefftS3)
plt.title("Spectre de phase de s3")
plt.xlabel('f (Hz)')
plt.ylabel('Phase (rad)')

# Affichage des spectres de s4
plt.subplot(4, 2, 7)
plt.stem(f, ampfftS4)
plt.title("Spectre d'amplitude de s4")
plt.xlabel('f (Hz)')
plt.ylabel('Amplitude')

plt.subplot(4, 2, 8)
plt.stem(f, phasefftS4)
plt.title("Spectre de phase de s4")
plt.xlabel('f (Hz)')
plt.ylabel('Phase (rad)')

plt.tight_layout()
plt.show()
