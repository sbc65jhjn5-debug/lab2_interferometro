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
    v_media = delta_s / DELTA_T # m/s
    sigma_v_rel = 0.02
    sigma_v     = sigma_v_rel * v_media  # m/s
    print(f"Velocita media: {v_media:.4f} ± {sigma_v} m/s")

    # I dati sono conteggi di fotoni, quindi in generale su I c'è sempre un'incertezza di Poisson
    sigma_poisson = np.sqrt(np.maximum(int_data, 1))  # sqrt(I), min 1 per evitare sqrt(0)

    # scaliamo la scala delle y (verticale) in modo da avere lo zero 
    # in corrispondenza dell'orizzontale del tavolo:

    y_scala = np.linspace (SPAZIO_INIZIO - zero, SPAZIO_FINE - zero, num=len(int_data))

    t = np.linspace(0, DELTA_T, len(int_data))
    sigma_y_i = sigma_v * t  # incertezza sulla posizione, cresce nel tempo [m]

    mask = y_scala > 0.03 # consideriamo solo i dati a destra dello zero, dove si vede meglio la diffrazione

    # ── Step 1: fit con solo errore di Poisson ────────────────────────────────

    ls_step1 = LeastSquares (y_scala,
                             int_data,
                             sigma_poisson,
                             I_teorica
                             )
    
    ls_step1.mask = mask
    
    results1 = []
    for N_prova in [2, 3, 4, 5, 6]:
        m1 = Minuit(ls_step1, I_0=50000, N=N_prova, a=0.00013, B=0, c=0, m=1)
        m1.fixed["N"] = True
        m1.migrad()
        results1.append((N_prova, m1.fval))  # fval = valore minimo del chi²
        print(f"N={N_prova}:  chi²={m1.fval:.1f},  a={m1.values['a']:.4e},  I0={m1.values['I_0']:.1f}")
    N_migliore = min(results1, key=lambda x: x[1])[0]
    print(f"\nN migliore: {N_migliore}")
    
    m_step1 = Minuit (ls_step1,
                      I_0 = 50000,
                      N = N_migliore,
                      a = 0.00013,
                      B = 500,
                      c = 0,
                      m = 1
                      )
    
    m_step1.fixed["N"] = True
    m_step1.limits["a"] = (0, 0.0002) # a deve essere positivo
    m_step1.limits["B"] = (1.3e4, None) # B deve essere positivo, e superiore a 1.3 * 10^4

    m_step1.migrad ()

    # CALCOLO ERRORE TOTALE
    y_fit_vals = I_teorica (y_scala, *m_step1.values)
    dI_dy = np.gradient (y_fit_vals, y_scala)

    sigma_tot = np.sqrt(sigma_poisson**2 + (dI_dy * sigma_y_i)**2) # Errore totale: Poisson + propagazione incertezza posizione


    # ── Step 2: fit con anche errore su y ────────────────────────────────

    ls_step2 = LeastSquares (y_scala,
                             int_data,
                             sigma_tot,
                             I_teorica
                             )
    
    ls_step2.mask = mask
    
    results2 = []
    for N_prova in [2, 3, 4, 5, 6]:
        m1 = Minuit(ls_step2, I_0=50000, N=N_prova, a=0.00013, B=0, c=0, m=1)
        m1.fixed["N"] = True
        m1.migrad()
        results2.append((N_prova, m1.fval))  # fval = valore minimo del chi²
        print(f"N={N_prova}:  chi²={m1.fval:.1f},  a={m1.values['a']:.4e},  I0={m1.values['I_0']:.1f}")
    N_migliore2 = min(results2, key=lambda x: x[1])[0]
    print(f"\nN migliore: {N_migliore}")
    
    m_step2 = Minuit (ls_step2,
                      I_0 = 50000,
                      N = N_migliore2,
                      a = 0.00018,
                      B = 500,
                      c = 0,
                      m = 1
                      )
    
    m_step2.fixed["N"] = True
    m_step2.limits["a"] = (0.0001, 0.0002) # a deve essere positivo
    m_step2.limits["B"] = (1.1e4, None) # B deve essere positivo, e superiore a 1.3 * 10^4

    m_step2.migrad ()


    I_0_fit, N_fit, a_fit, B_fit, c_fit, m_fit = m_step2.values["I_0"], m_step2.values["N"], m_step2.values["a"], m_step2.values["B"], m_step2.values["c"], m_step2.values["m"]
    for par, val, err in zip (m_step2.parameters, m_step2.values, m_step2.errors):
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
    

    # banda di errore dei dati sperimentali
    ax.fill_between(y_scala,
                    int_data - sigma_tot,
                    int_data + sigma_tot,
                    color='lightblue', alpha=0.7, label="Errore totale (Poisson + posizione)")
    
    
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