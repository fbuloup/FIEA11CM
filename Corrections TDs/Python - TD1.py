# =============================================================================
# TD1 - TRAITEMENT DE SIGNAL
# =============================================================================

# Import des librairies nécessaires
import matplotlib.pyplot as plt
import numpy as np

#%% 

# =============================================================================
#   EXERCICE 2
# =============================================================================

# PREMIERE PARTIE

plt.close('all')

# On crée un vecteur pour faire 4 périodes 
A = np.arange(0,8*np.pi,8*np.pi/100)

# On utilise les fonctions Sin et Cos
Sin = np.sin(A)
Cos = np.cos(A)

# On crée la figure pour vsualiser les fonctions Cos et Sin

plt.figure(1)
plt.plot(A,Sin, color = 'blue', label = 'Sinus')
plt.plot(A,Cos, color = 'red', label = 'Cosinus')
plt.legend(loc = 'upper left')


# DEUXIEME PARTIE

# On pose θ = wt avec w = 2.pi.f

# On crée un vecteur temps d'une seconde à 100Hz
t = np.arange(0,1,1/100)

# On crée les 3 fréquences demandées
freq1 = 1
freq2 = 2
freq3 = 5

# On crée les fonctions Sin associées aux 3 fréquences
fonc1 = np.sin(2*np.pi*freq1*t)
fonc2 = np.sin(2*np.pi*freq2*t)
fonc3 = np.sin(2*np.pi*freq3*t)

# On visualise les 3 fonctions
plt.figure(2)
plt.plot(t,fonc1, color = 'blue', linestyle = 'solid', label = '1Hz')
plt.plot(t,fonc2, color = 'red', linestyle = 'dashed', label = '2Hz')
plt.plot(t,fonc3, color = 'green', linestyle = 'dotted', label = '5Hz')
plt.legend(loc = 'upper left')

