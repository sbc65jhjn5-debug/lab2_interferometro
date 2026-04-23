import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HeNe = 632.8e-7  # lunghezza d'onda in cm
d = 0.1  # distanza tra le fenditure in cm
L = 97.5  # distanza tra le fenditure e lo schermo in cm
v_media = (14.0 - 6.0) / 22.4  # velocità media in cm/s

# La formula teorica per l'intensità è:
# I(x) = offset + slope * cos^2(k * (x - x0))
# dove k = 2 * pi * d / (lambda * L)
k_teorico = 2 * np.pi * d / (HeNe * L)

def theoretical_intensity (x, offset, slope, k, x0):
    # Modello con envelope gaussiana
    I = offset + slope * np.cos(k * (x - x0)) ** 2
    return I

def Irma (x, L_0, lamb, D):
    return (d * (np.sin(np.arctan(D/L_0)) - np.sin(np.arctan(D / abs(x - 4)))) / lamb)**2

data = pd.read_csv("run_continua.txt")
y_data = data.iloc[:, 0].to_numpy(dtype=float)

x_axis = np.linspace(0, v_media * 22.4, len(y_data))  # spazio in cm

fig, ax = plt.subplots()

ax.set_title("LDR data")
ax.set_xlabel("Space (cm)")
ax.set_ylabel("LDR value")

ax.plot(x_axis, y_data, label="LDR", marker = '', color='blue', linewidth=1)
ax.plot (x_axis,
         [theoretical_intensity(x, 0, 50000, k_teorico, 1) for x in x_axis],
         label="Teorico 1", color='red'
         )
'''
ax.plot (x_axis,
         [Irma(x, L, HeNe, d) for x in x_axis],
         label="Teorico 2", color='magenta'
         )
'''
plt.legend ()
plt.grid (True)
plt.show ()