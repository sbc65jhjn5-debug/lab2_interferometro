import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.signal import find_peaks, savgol_filter

DELTA_T = 22.4
SPAZIO_INIZIO_CM = 6.0
SPAZIO_FINE_CM = 14.0

def delta (theta, lamb = 632.8e-9, d):
    # theta è angolo di osservazione
    return (2 * np.pi / lamb) * d * np.sin (theta)

def z (theta, lamb = 632.8e-9, a = 0.0001):
    # theta è angolo di osservazione
    # a è la larghezza della singola fenditura
    return (np.pi * a / lamb) * np.sin (theta)

def I_teorica (delta, N, z):
    return I_0 * (np.sin (N * delta / 2) / (N * np.sin (delta / 2)))**2 * (np.sin (z) / z)**2
    # al denominatore del primo fattore è dubbio se ci sia o no N (in appunti c'è ma a voce non c'era...)


if __name__ == "__main__":

    data = pd.read_csv("run_continua.txt")
    y_data = data.iloc[:, 0].to_numpy(dtype=float)

    delta_s = SPAZIO_FINE_CM - SPAZIO_INIZIO_CM
    v_media = delta_s / DELTA_T
    print(f"Velocita media: {v_media:.4f} cm/s")

