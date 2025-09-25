#### Ce script contient les deux méthodes de calcul et d'affichage des
#### spectres : monolatéral et bilatéral

## Import des bilbiothèques utilisées
import numpy as np
import matplotlib.pyplot as plt

# Fermer toutes les fenêtres graphiques
plt.close('all')

# Effacer les varaibles
from IPython import get_ipython;   
get_ipython().magic('reset -fs') # s : history, f : force (no question)

# Création du vecteur temporel
Fe = 1000
Te = 1/Fe
nbSamples = 3000 # Tester avec e.g. 3003
t = np.arange(0, nbSamples*Te, Te)

# Synthèse des signaux
s1 = 3*np.ones(nbSamples)
s2 = 6*np.sin(2 * np.pi * t)
s3 = 1 + 2*np.cos(10 * np.pi * t) + 4*np.cos(20 * np.pi * t)
s4 = np.cos(100 * np.pi * t) + np.cos(100 * np.pi * t) * np.cos(100 * np.pi * t)

#%%#####################
# Spectres monolatéraux
#######################
# Calcul FFT
fftS1 = np.fft.fft(s1)/nbSamples
fftS2 = np.fft.fft(s2)/nbSamples
fftS3 = np.fft.fft(s3)/nbSamples
fftS4 = np.fft.fft(s4)/nbSamples

# Module
moduleFFTS1 = np.abs(fftS1)
moduleFFTS2 = np.abs(fftS2)
moduleFFTS3 = np.abs(fftS3)
moduleFFTS4 = np.abs(fftS4)

# Phase
phaseFFTS1 = np.angle(fftS1)
phaseFFTS2 = np.angle(fftS2)
phaseFFTS3 = np.angle(fftS3)
phaseFFTS4 = np.angle(fftS4)

# Nettoyage des phases
epsilon = 1e-1
phaseFFTS1[moduleFFTS1 < epsilon] = 0
phaseFFTS2[moduleFFTS2 < epsilon] = 0
phaseFFTS3[moduleFFTS3 < epsilon] = 0
phaseFFTS4[moduleFFTS4 < epsilon] = 0

# Création du vecteur fréquentiel (monolatéral)
df = Fe/nbSamples
if nbSamples % 2 == 1:
    # nbSamples impair
    N = (nbSamples + 1)//2
    f = np.arange(0, Fe/2 + df/2, df )
else:
    # nbSamples pair
    N = nbSamples//2
    f = np.arange(0, Fe/2, df )

# Arrangement pour obtenir les bonnes amplitudes (méthode 1)
moduleFFTS1 = np.concatenate((moduleFFTS1[0], 2*moduleFFTS1[1:N]), axis = None)
moduleFFTS2 = np.concatenate((moduleFFTS2[0], 2*moduleFFTS2[1:N]), axis = None)
moduleFFTS3 = np.concatenate((moduleFFTS3[0], 2*moduleFFTS3[1:N]), axis = None)
moduleFFTS4 = np.concatenate((moduleFFTS4[0], 2*moduleFFTS4[1:N]), axis = None)
# Arrangement pour obtenir les bonnes amplitudes (méthode 2)
# moduleFFTS1[1:N] =  2*moduleFFTS1[1:N] 
# moduleFFTS2[1:N] =  2*moduleFFTS2[1:N] 
# moduleFFTS3[1:N] =  2*moduleFFTS3[1:N] 
# moduleFFTS4[1:N] =  2*moduleFFTS4[1:N] 
# moduleFFTS1 = moduleFFTS1[0:N] 
# moduleFFTS2 = moduleFFTS2[0:N] 
# moduleFFTS3 = moduleFFTS3[0:N] 
# moduleFFTS4 = moduleFFTS4[0:N] 

phaseFFTS1 = phaseFFTS1[0:N] 
phaseFFTS2 = phaseFFTS2[0:N] 
phaseFFTS3 = phaseFFTS3[0:N] 
phaseFFTS4 = phaseFFTS4[0:N] 

# Affichage
plt.figure(1)

plt.subplot(4,2,1)
plt.stem(f, moduleFFTS1)
plt.title("Spectre amplitude de s1")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,2,2)
plt.stem(f, phaseFFTS1)
plt.title("Spectre de phase de s1")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude (rad)")
plt.grid()

plt.subplot(4,2,3)
plt.stem(f, moduleFFTS2)
plt.title("Spectre amplitude de s2")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,2,4)
plt.stem(f, phaseFFTS2)
plt.title("Spectre de phase de s2")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude (rad)")
plt.grid()

plt.subplot(4,2,5)
plt.stem(f, moduleFFTS3)
plt.title("Spectre amplitude de s3")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,2,6)
plt.stem(f, phaseFFTS3)
plt.title("Spectre de phase de s3")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude (rad)")
plt.grid()

plt.subplot(4,2,7)
plt.stem(f, moduleFFTS4)
plt.title("Spectre amplitude de s4")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,2,8)
plt.stem(f, phaseFFTS4)
plt.title("Spectre de phase de s4")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude (rad)")
plt.grid()

plt.tight_layout()

#%%####################
# Spectres bilatéraux
#####################
# on utilise fft PUIS fftshift : il faut utiliser les deux
fftS1 = np.fft.fftshift(np.fft.fft(s1)/nbSamples)
fftS2 = np.fft.fftshift(np.fft.fft(s2)/nbSamples)
fftS3 = np.fft.fftshift(np.fft.fft(s3)/nbSamples)
fftS4 = np.fft.fftshift( np.fft.fft(s4)/nbSamples)

# Modules
moduleFFTS1 = np.abs(fftS1)
moduleFFTS2 = np.abs(fftS2)
moduleFFTS3 = np.abs(fftS3)
moduleFFTS4 = np.abs(fftS4)

# Phases
phaseFFTS1 = np.angle(fftS1)
phaseFFTS2 = np.angle(fftS2)
phaseFFTS3 = np.angle(fftS3)
phaseFFTS4 = np.angle(fftS4)

# Nettoyage des phases
epsilon = 1e-1
phaseFFTS1[moduleFFTS1 < epsilon] = 0
phaseFFTS2[moduleFFTS2 < epsilon] = 0
phaseFFTS3[moduleFFTS3 < epsilon] = 0
phaseFFTS4[moduleFFTS4 < epsilon] = 0

# Création du vecteur fréquentiel (bilatéral)
df = Fe/nbSamples
if nbSamples % 2 == 1:
    f = np.arange(-Fe/2 + df/2, Fe/2 + df/2, df )
else:
    f = np.arange(-Fe/2, Fe/2, df )
    
# Affichage
plt.figure(2)

plt.subplot(4,2,1)
plt.stem(f, moduleFFTS1)
plt.title("Spectre amplitude de s1")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,2,2)
plt.stem(f, phaseFFTS1)
plt.title("Spectre de phase de s1")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude (rad)")
plt.grid()

plt.subplot(4,2,3)
plt.stem(f, moduleFFTS2)
plt.title("Spectre amplitude de s2")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,2,4)
plt.stem(f, phaseFFTS2)
plt.title("Spectre de phase de s2")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude (rad)")
plt.grid()

plt.subplot(4,2,5)
plt.stem(f, moduleFFTS3)
plt.title("Spectre amplitude de s3")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,2,6)
plt.stem(f, phaseFFTS3)
plt.title("Spectre de phase de s3")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude (rad)")
plt.grid()

plt.subplot(4,2,7)
plt.stem(f, moduleFFTS4)
plt.title("Spectre amplitude de s4")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4,2,8)
plt.stem(f, phaseFFTS4)
plt.title("Spectre de phase de s4")
plt.xlabel("f (Hz)")
plt.ylabel("Amplitude (rad)")
plt.grid()