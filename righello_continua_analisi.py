import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares

DELTA_T = 22.4
SPAZIO_INIZIO = 0.06 # m
SPAZIO_FINE = 0.14 # m

lambda_laser = 632.8e-9 # lunghezza d'onda del laser in metri

d = 0.0005 # passo del righello (m)

zero = 0.0525 # valore ottenuto da "righello_mano" (m)

D = np.mean (np.array([0.974, 0.975, 0.974, 0.975, 0.973, 0.974])) # m
sigma_D = np.std (np.array([0.974, 0.975, 0.974, 0.975, 0.973, 0.974])) / np.sqrt(6)


def I_teorica (y, I_0, N, a, B, c, m):

    delta = (2 * np.pi / lambda_laser) * d * (np.cos (np.arctan (zero / D)) - np.cos (np.arctan (((y/m) + c) / D)))
    z = (np.pi / lambda_laser) * a * (np.cos (np.arctan (zero / D)) - np.cos (np.arctan (((y/m) + c) / D)))

    return I_0 * (np.sin (N * delta / 2) / (N * np.sin (delta / 2)))**2 * (np.sin (z) / z)**2 + B
    # al denominatore del primo fattore è dubbio se ci sia o no N (in appunti c'è ma a voce non c'era...)


if __name__ == "__main__":

    data = pd.read_csv("run_continua.txt")
    int_data = data.iloc[:, 0].to_numpy(dtype=float)

    int_err = np.ones (len(int_data)) * 2000

    delta_s = SPAZIO_FINE - SPAZIO_INIZIO
    v_media = delta_s / DELTA_T
    print(f"Velocita media: {v_media:.4f} m/s")

    # scaliamo la scala delle y (verticale) in modo da avere lo zero 
    # in corrispondenza dell'orizzontale del tavolo:

    y_scala = np.linspace (SPAZIO_INIZIO - zero, SPAZIO_FINE - zero, num=len(int_data))

    mask = y_scala > 0.03 # consideriamo solo i dati a destra dello zero, dove si vede meglio la diffrazione

    ls = LeastSquares (y_scala,
                       int_data,
                       int_err,
                       I_teorica
                       )
    
    ls.mask = mask
    
    results = []
    for N_prova in [2, 3, 4, 5, 6]:
        m1 = Minuit(ls, I_0=50000, N=N_prova, a=0.00013, B=0, c=0, m=1)
        m1.fixed["N"] = True
        m1.migrad()
        results.append((N_prova, m1.fval))  # fval = valore minimo del chi²
        print(f"N={N_prova}:  chi²={m1.fval:.1f},  a={m1.values['a']:.4e},  I0={m1.values['I_0']:.1f}")
    N_migliore = min(results, key=lambda x: x[1])[0]
    print(f"\nN migliore: {N_migliore}")
    
    m = Minuit (ls,
                I_0 = 50000,
                N = N_migliore,
                a = 0.00013,
                B = 500,
                c = 0,
                m = 1
                )
    
    m.fixed["N"] = True
    m.limits["a"] = (0, 0.0002) # a deve essere positivo
    m.limits["B"] = (1.3e4, None) # B deve essere positivo, e superiore a 1.3 * 10^4
    #m.limits["I_0"] = (3.9e4, None) # I_0 deve essere positivo e superiore a 45000
    m.migrad ()

    I_0_fit, N_fit, a_fit, B_fit, c_fit, m_fit = m.values["I_0"], m.values["N"], m.values["a"], m.values["B"], m.values["c"], m.values["m"]
    for par, val, err in zip (m.parameters, m.values, m.errors):
        print(f"{par}: {val:.6e} ± {err:.6e}")
    
    fig, ax = plt.subplots()

    ax.set_title ("Intensità in funzione della posizione sullo schermo")
    ax.set_xlabel ("Posizione (m)")
    ax.set_ylabel ("Intensità (a.u.)")

    ax.plot (y_scala, 
             int_data, 
             marker = '',
             color = 'blue',
             linewidth = 1, 
             label="Dati sperimentali")
    
    # area che indica l'errore dei dati sperimentali
    ax.fill_between (y_scala, int_data - int_err, int_data + int_err, color='lightblue', alpha=0.7, label="Errore dei dati sperimentali")
    
    
    ax.plot (y_scala,
             I_teorica (y_scala, I_0_fit, N_fit, a_fit, B_fit, c_fit, m_fit),
             marker = '',
             color = 'red',
             label="Fit teorico")
    
    # quadrato per mask
    ax.fill_between (y_scala, 0, 57000, where=mask, color='mistyrose', alpha=0.6, label="Dati usati per il fit")
    
    plt.legend (loc="lower right")
    plt.grid (True)
    plt.show ()