import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal import filtfilt
from scipy.signal import butter
from scipy.io.wavfile import read
import sounddevice as sd

#%% EXO 1

# Définition du vecteur temporel
Fe = 4
Te = 1 / Fe
t = np.arange(0, 10 + Te, Te)

# Définition du signal
s = 3 + np.sin(2 * np.pi * t)

# Tracé du signal
plt.figure(figsize=(8, 10))

plt.subplot(3, 1, 1)
plt.plot(t, s)
plt.title('3 + sin(2 * pi * t)')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')

# Filtre Moyenneur sur quatre points
b = 1 / 4 * np.array([1, 1, 1, 1])
a = [1]
s_filtered = lfilter(b, a, s)

plt.subplot(3, 1, 2)
plt.plot(t, s_filtered)
plt.title('s_Filtered')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')

# Filtre dérivateur d'ordre un
b = 1 / Te * np.array([1, -1])
a = [1]
s_derive = lfilter(b, a, s)

plt.subplot(3, 1, 3)
plt.plot(t, s_derive, label='s_derive')

# Calcul de la dérivée théorique
s_derive_theo = 2 * np.pi * np.cos(2 * np.pi * t)
plt.plot(t, s_derive_theo, 'r', label='s_derive_theo')

plt.title('s_derive et s_derive_theo')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()


#%% EXO 2

Audio_path1 = 'C:/Users/Egiziano/Documents/THESE/ENSEIGNEMENT/Traitement de Signal/Christmas_song.wav'

Audio_path2 = 'C:/Users/Egiziano/Documents/THESE/ENSEIGNEMENT/Traitement de Signal/Christmas_noise.wav'

Fe, signal = read(Audio_path1)  # Lecture du signal
Fe, signal_bruit = read(Audio_path2)  # Lecture du signal

signal = signal / np.max(np.abs(signal))
signal_bruit = signal_bruit / np.max(np.abs(signal_bruit))

# sd.play(signal, Fe)
# sd.wait()


N = len(signal)
df = Fe / N
T = np.arange(0, N*1/Fe, 1/Fe)

fftfreq =  np.fft.fftfreq(N, 1/Fe)
fftfreqmono = fftfreq[fftfreq>=0]

fftSignal = np.fft.fft(signal) / N

# Calcul du module
magfftSignal = np.abs(fftSignal)
magfftSignal = np.concatenate((magfftSignal[0], 2 * magfftSignal[1:N//2]), axis = None)


plt.figure(1)

# Affichage du module
plt.stem(fftfreqmono, magfftSignal, linefmt = 'grey')
plt.title('Spectre de module de la note')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')


plt.figure(2)

# Comparaison des deux signaux
plt.subplot(2,1,1)
plt.plot(T, signal, color = 'grey')
plt.title('Signal clair')
plt.xlabel('Temps en s')
plt.ylabel('Signal')

plt.subplot(2,1,2)
plt.plot(T, signal_bruit, color = 'red')
plt.title('Signal bruité')
plt.xlabel('Temps en s')
plt.ylabel('Signal')


# On écoute le signal bruité
# sd.play(signal_bruit, Fe)
# sd.wait()


# On refait la même chose avec le signal bruité
fftSignal_bruit = np.fft.fft(signal_bruit) / N

# Calcul du module
magfftSignal_bruit = np.abs(fftSignal_bruit)
magfftSignal_bruit = np.concatenate((magfftSignal_bruit[0], 2 * magfftSignal_bruit[1:N//2]), axis = None)


plt.figure(3)

# Affichage du module
plt.stem(fftfreqmono, magfftSignal_bruit, linefmt = 'red')
plt.title('Spectre de module de la note bruitée')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')


# Création du filtre buttterworth passe bas d'ordre 4 et de fréquence de coupure 800
Fcoup  = 2500
b1, a1 = butter(4,Fcoup/(Fe/2), 'low')
signal_filtre = filtfilt(b1, a1, signal_bruit)

signal_filtre = signal_filtre / np.max(np.abs(signal_filtre))



plt.figure(4)

# Comparaison des deux signaux
plt.subplot(3,1,1)
plt.plot(T, signal, color = 'grey')
plt.title('Signal clair')
plt.xlabel('Temps en s')
plt.ylabel('Signal')

plt.subplot(3,1,2)
plt.plot(T, signal_bruit, color = 'red')
plt.title('Signal bruité')
plt.xlabel('Temps en s')
plt.ylabel('Signal')

plt.subplot(3,1,3)
plt.plot(T, signal_filtre, color = 'blue')
plt.title('Signal bruité')
plt.xlabel('Temps en s')
plt.ylabel('Signal')


# On refait la même chose avec le signal filtré
fftSignal_filtre = np.fft.fft(signal_filtre) / N

# Calcul du module
magfftSignal_filtre = np.abs(fftSignal_filtre)
magfftSignal_filtre = np.concatenate((magfftSignal_filtre[0], 2 * magfftSignal_filtre[1:N//2]), axis = None)


plt.figure(5)

# Affichage du module
plt.stem(fftfreqmono, magfftSignal_filtre, linefmt = 'blue')
plt.title('Spectre de module de la note filtrée')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')


#%%

#On écoute les différents signaux
sd.play(signal, Fe)
sd.wait()

sd.play(signal_bruit, Fe)
sd.wait()

sd.play(signal_filtre, Fe)
sd.wait()
