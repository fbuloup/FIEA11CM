# Imports nécessaires
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal import butter
from scipy.io.wavfile import read
import sounddevice as sd

# I.1 Lecture du fichier de données
fs, signal = read('13918_AV.wav')
# Normalisation des amplitudes dans l'intervallez [-1, 1]
# Python ne se comporte pas exactement comme Matlab ici...
signal = signal / np.max(signal)

# I.2 Création du vecteur temporel
nbSamples = len(signal)
Ts = 1/fs
t = np.arange(0, nbSamples*Ts, Ts)

# I.3 Affichage du signal
plt.plot(t, signal);

## I.4 Titre et axes
plt.title('Signal signal.wav');
plt.xlabel('temps (s)');
plt.ylabel('Amplitude');

# I.5 Calcul FFT
fftSignal = np.fft.fftshift(np.fft.fft(signal)/nbSamples);

# I.6 Calcul module et phase FFT
magFFTSignal = np.abs(fftSignal);
phaseFFTSignal = np.angle(fftSignal);

# I.7 Création du vecteur fréquentiel (bilatéral)
df = fs/nbSamples
if nbSamples % 2 == 0:
    # nbSamples pair
    f = np.arange(-fs/2, fs/2, df)
else:
    # nbSamples impair
    f = np.arange(-fs/2 + df/2, fs/2 + df/2, df)

## I.8 Affichage module FFT
plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(f, magFFTSignal)
plt.title("Spectre bilatéral d'amplitude du signal")
plt.xlabel('fréquence (Hz)')
plt.ylabel('Amplitude');

# I.9 Affichage phase FFT
plt.subplot(2, 1, 2);
plt.plot(f, phaseFFTSignal);
plt.title('Spectre bilatéral de phase du signal');
plt.xlabel('fréquence (Hz)');
plt.ylabel('Phase en rad');

# I.10 Affichage phase FFT
sd.play(signal, fs)
sd.wait()
# Conversation et respiration à supprimer...

# II.1 Filtre récursif ?
# Il s'agit d'un filtre récursif puisque des échantillons passés de
# la sortie sont utilisés pour calculer la sortie à l'instant courant

# II.2 Calcul de la TZ de l'?quation aux diff?rences
# S(z) + a1.z^(-1) S(z) + a2.z^(-2) S(z) + a3.z^(-3) S(z) + a4.z^(-4) S(z)  = b1.E(z) + b2.z^(-1) E(z) + b3.z^(-2) E(z) + b4.z^(-3) E(z) + b5.z^(-4) E(z)

# II.3 Expression de la fonction de transfert en z
# H(z) = S(z)/E(z) = (b1 + b2.z^(-1) + b3.z^(-2) + b4.z^(-3) + b5.z^(-4)) / (1 + a1.z^(-1) + a2.z^(-2) + a3.z^(-3) + a4.z^(-4))

# II.4 Les vecteurs A et B pour la fonction lfilter sont donc de la forme
# A = [1, a1, a2, a3, a4]
# B = [b1, b2, b3, b4, b5]

# II.5 Filtrage du son
# Les vecteurs des coefficients A et B sont calculés avec la fonction butter
[B, A] = butter(4, 120/(fs/2), btype = 'low', analog=False)
# Le filtrage est effectué avec la fonction lfilter
filteredSignal = lfilter(B, A, signal);

# II.6 Affichage du signal filtré
plt.figure(3)
plt.plot(t, filteredSignal)
plt.title('Signal filtré')
plt.xlabel('temps (s)')
plt.ylabel('Amplitude');

# II.7 Spectres monolatéraux module et phase signal filtré
fftFilteredSignal = np.fft.fft(filteredSignal)/nbSamples;
magFFTFilteredSignal = np.abs(fftFilteredSignal);
phaseFFTFilteredSignal = np.angle(fftFilteredSignal);
if nbSamples % 2 == 0:
    # nbSamples pair
    N = nbSamples//2
    f = np.arange(0, fs/2, df)
else:
    # nbSamples impair
    N = (nbSamples + 1)//2
    f = np.arange(0, fs/2 + df/2, df)
plt.figure(4);
plt.subplot(2, 1, 1);
plt.plot(f, np.concatenate((magFFTFilteredSignal[0], 2*magFFTFilteredSignal[1:N]), axis = None))
plt.title("Spectre monolatéral d'amplitude du signal filtré");
plt.xlabel('fréquence (Hz)');
plt.ylabel('Amplitude');
plt.subplot(2, 1, 2);
plt.plot(f, phaseFFTFilteredSignal[0:N]);
plt.title('Spectre monolatéral de phase du signal filtré');
plt.xlabel('fréquence (Hz)');
plt.ylabel('Phase en rad');

# II.8 Résultat
sd.play(filteredSignal, fs)
sd.wait()
from scipy.signal import freqz
fbis, h = freqz(B, A, worN = nbSamples, fs = fs);
plt.figure(5)
plt.plot(fbis, np.abs(h));
plt.title('Réponse en fréquence du filtre');
plt.xlabel('f Hz');
plt.ylabel('Amplitude');
plt.show()
# à l'écoute, les deux parasites ont été supprimés et on remarque que la réponse en
# fréquence du filtre correspond bien à un passe bas dont la fréquence de
# coupure à 120Hz correspond bien à un gain de 1/sqrt(2). Les fréquences
# supérieures à 120Hz sont ""quasiment" toutes supprimées du signal original.
