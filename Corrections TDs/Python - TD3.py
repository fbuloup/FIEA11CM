# =============================================================================
# TD3 - TRAITEMENT DE SIGNAL
# =============================================================================

# Import des librairies nécessaires
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import sounddevice as sd

#%%
# =============================================================================
# EXERCICE 1 - Exemple - Discrétisation
# =============================================================================

# On définit les paramètres

A = 10
w = 5
n = 1000
T = 2*np.pi/w

# Définition du tableau Te
Te = np.array([1, 0.5, 0.1, 0.01, 0.001])

# Génération de la colormap 'winter' avec le même nombre de couleurs que la longueur de Te
color0 = plt.cm.winter(np.linspace(0, 1, len(Te)))


plt.figure(1)

# Boucle sur les différentes périodes d'échantillonnage
for i, te in enumerate(Te):
    t0 = np.arange(0, n * te, te)
    cos_ech0 = A * np.cos(w * te * np.arange(0, n))
    
    # Subplot 1
    plt.subplot(3, 1, 1)
    plt.plot(t0, cos_ech0, '+-', color=color0[i], label=f'T = {te}')
    plt.xlabel('t0')
    plt.ylabel('Amplitude')
    plt.title('Signal sinusoïdal discrétisé')
    plt.axis([0, 1000, -10, 10])
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # Subplot 2
    plt.subplot(3, 1, 2)
    plt.plot(cos_ech0, '+-', color=color0[i], label=f'T = {te}')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.axis([0, 1200, -10, 10])
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # Subplot 3
    plt.subplot(3, 1, 3)
    plt.plot(t0, cos_ech0, '+-', color=color0[i], label=f'T = {te}')
    plt.xlabel('t0')
    plt.ylabel('Amplitude')
    plt.axis([0, 6, -10, 10])
    plt.grid(True)
    plt.legend(loc='upper right')

# Ajuster l'espacement entre les subplots
plt.tight_layout()

# Afficher la figure
plt.show()

# Reflexion commune : Parmi les différents Te proposés lequel est le plus opti et pourquoi ?

#%%

# =============================================================================
# EXERCICE 2 - Application sous Python
# =============================================================================

# On définit l'emplacement du fichier
Audio_path = 'C:/Users/buloup/Desktop/Traitement de Signal - Python/mi2_bute_ros.wav'

# On importe le fichier ainsi que sa fréquence d'échantillonage
Fs, y = wavfile.read(Audio_path)


# Lecture du fichier audio à plusieurs fréquences
sd.play(y, Fs)
sd.wait() 

sd.play(y, Fs*2)
sd.wait() 

sd.play(y, Fs/2)
sd.wait()


# Création d'un vecteur temps
Tps = np.arange(0, len(y)/Fs, 1/Fs,)
Tps_2 = np.arange(0, len(y)/(Fs*2), 1/(Fs*2))

# On trace le signal au cours du temps
plt.figure(2)

plt.subplot(2,1,1)
plt.plot(Tps, y, '+-', color = 'navy')

plt.subplot(2,1,2)
plt.plot(Tps_2, y, '+-', color = 'navy')


#%%

# =============================================================================
#  EXERCICE 3 - Signaux élémentaires
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
Te = 1
Tdebut = -10
Tfin = 10
time = np.arange(Tdebut, Tfin, Te)

# Impulsion unitaire
T_imp = 3
unitImpulse = np.zeros(time.shape)
I_imp = np.where(time == T_imp)[0]
if I_imp.size > 0:
    unitImpulse[I_imp[0]] = 1

# Fonction échelon
T_Ech = -3
step = np.zeros(time.shape)
I_imp = np.where(time >= T_Ech)[0]
step[I_imp] = 1

# Fonction rectangulaire (façon brute)
N = 6
rect = np.concatenate([np.zeros(7), np.ones(N), np.zeros(20-N-7)])

# Fonction rectangulaire (façon classe)
T_Ech_d = -3
T_Ech_f = 2
rect = np.zeros(time.shape)
I_deb = np.where(time == T_Ech_d)[0]
I_fin = np.where(time == T_Ech_f)[0]
if I_deb.size > 0 and I_fin.size > 0:
    rect[I_deb[0]:I_fin[0]+1] = 1

# Affichage normal
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(time, unitImpulse, '-xr')
plt.xlabel('time')
plt.ylabel('Unit impulse function')
plt.title('Affichage normal')

plt.subplot(3,1,2)
plt.plot(time, step, '-xb')
plt.xlabel('time')
plt.ylabel('Step function')

plt.subplot(3,1,3)
plt.plot(time, rect, '-xm')
plt.xlabel('time')
plt.ylabel('Rectangular function')

# Sinus et cosinus discrets (affichage normal)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(time, np.sin(2 * np.pi / 20 * time), '-xr')
plt.xlabel('time')
plt.ylabel('sinus')
plt.title('Sinus et cosinus discrets')

plt.subplot(2,1,2)
plt.plot(time, np.cos(2 * np.pi / 20 * time), '-xb')
plt.xlabel('time')
plt.ylabel('cosinus')

# Affichage séquence discrète
plt.figure(3)
plt.subplot(3,1,1)
plt.stem(time, unitImpulse, 'r')
plt.xlabel('time')
plt.ylabel('Unit impulse function')
plt.title('Affichage séquence discrète')

plt.subplot(3,1,2)
plt.stem(time, step, 'b')
plt.xlabel('time')
plt.ylabel('Step function')

plt.subplot(3,1,3)
plt.stem(time, rect, 'm')
plt.xlabel('time')
plt.ylabel('Rectangular function')

# Sinus et cosinus discrets (affichage séquence discrète)
plt.figure(4)
plt.subplot(2,1,1)
plt.stem(time, np.sin(2 * np.pi / 20 * time), 'r')
plt.xlabel('time')
plt.ylabel('sinus')
plt.title('Sinus et cosinus discrets')

plt.subplot(2,1,2)
plt.stem(time, np.cos(2 * np.pi / 20 * time), 'b')
plt.xlabel('time')
plt.ylabel('cosinus')

plt.show()


#%% 

# =============================================================================
# EXERCICE 4 - Transition vers l'aspect fréquentiel
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


# Création du vecteur temporel
t = np.arange(0, 2, 0.001)

# Définition du signal s
s = (4/np.pi)*(np.sin(2*np.pi*t) + 1/3*np.sin(2*np.pi*3*t) + 1/5*np.sin(2*np.pi*5*t))

# On trace le signal

plt.figure(1)
plt.plot(t,s, '-r')
plt.xlabel('Temps')
plt.ylabel('s(t)')

# On fait la somme demandée
Sum = np.zeros(t.shape)

plt.figure(2)

for k in np.arange(0,49):
    
    y = (4/(2*k + 1))*np.sin(2*np.pi*(2*k + 1)*t)
    
    Sum += y
    
    plt.plot(t,Sum)
    plt.ylabel('Fonction s(t) en fonction des harmoniques')
    plt.xlabel('Temps (s)')
    plt.pause(0.15)


