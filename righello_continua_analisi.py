import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.signal import find_peaks, savgol_filter

DELTA_T = 22.4
SPAZIO_INIZIO_CM = 6.0
SPAZIO_FINE_CM = 14.0

if __name__ == "__main__":
    
    data = pd.read_csv("run_continua.txt")
    y_data = data.iloc[:, 0].to_numpy(dtype=float)

    delta_s = SPAZIO_FINE_CM - SPAZIO_INIZIO_CM
    v_media = delta_s / DELTA_T
    print(f"Velocita media: {v_media:.4f} cm/s")