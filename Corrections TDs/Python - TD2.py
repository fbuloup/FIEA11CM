# =============================================================================
# TD2 - TRAITEMENT DE SIGNAL
# =============================================================================

# Import des librairies nécessaires
import matplotlib.pyplot as plt
import numpy as np

#%%
# =============================================================================
# EXERCICE 2 - Complexes avec Python
# =============================================================================

# 1
z1 = 2 + 5j

z2 = 10*np.exp(1j*np.pi/4)

z3 = 4

z4 = 5j


# 3
Mod = np.abs(z1)
Arg = np.angle(z1)

print(f'Module: {Mod}')
print(f'Argument : (angle): {Arg} radians')

# 4
Real = np.real(z2)
Imag = np.imag(z2)

print(f'Partie réelle: {Real}')
print(f'Partie imaginaire: {Imag}')

# 5
z_1 = np.conj(z1)
z_2 = np.conj(z2)
z_3 = np.conj(z3)
z_4 = np.conj(z4)

print(f'Conjugué de z1: {z_1}')
print(f'Conjugué de z2: {z_2}')
print(f'Conjugué de z3: {z_3}')
print(f'Conjugué de z4: {z_4}')

print(f'z1*z2 = {z1*z2}')
print(f'z1/z2 = {z1/z2}')
print(f'z3*z4 = {z3*z4}')
print(f'z1/z4 = {z1/z4}')

# 6
plt.figure(1)

plt.scatter(np.real(z1), np.imag(z1), color = 'red', marker = 'o')
plt.scatter(np.real(z2), np.imag(z2), color = 'blue', marker = '*')
plt.scatter(z3, 0, color = 'orange', marker = '+')
plt.scatter(0, np.imag(z4), color = 'green', marker = 'x')

plt.xlabel('Partie Réelle')
plt.ylabel('Partie Imaginaire')

plt.xlim([-10, 10])
plt.ylim([-10, 10])

plt.axhline(y = 0, linestyle = 'dotted', color = 'k')
plt.axvline(x = 0, linestyle = 'dotted', color = 'k')


