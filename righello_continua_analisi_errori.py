import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares

DELTA_T = 22.4
SPAZIO_INIZIO = 0.06  # m
SPAZIO_FINE   = 0.14  # m

lambda_laser = 632.8e-9  # lunghezza d'onda del laser in metri
d = 0.0005               # passo del righello (m)

D = np.mean(np.array([0.974, 0.975, 0.974, 0.975, 0.973, 0.974]))  # m
sigma_D = np.std(np.array([0.974, 0.975, 0.974, 0.975, 0.973, 0.974])) / np.sqrt(6)

# ── MODIFICA 1: zero non è più hardcoded qui, viene calcolato sotto ──────────
# zero = 0.0525  <-- rimosso, ora viene calcolato dai dati

def I_teorica(y, I_0, N, a, B, c, m_par, zero_val):
    """
    Aggiunto zero_val come parametro esplicito invece di variabile globale.
    Questo evita il bug della variabile globale usata prima della definizione.
    """
    eff_y = (y / m_par) + c
    cos0  = np.cos(np.arctan(zero_val / D))
    cosy  = np.cos(np.arctan(eff_y / D))

    delta = (2 * np.pi / lambda_laser) * d * (cos0 - cosy)
    z     = (np.pi  / lambda_laser) * a * (cos0 - cosy)

    with np.errstate(invalid='ignore', divide='ignore'):
        grating = np.where(
            np.abs(np.sin(delta / 2)) < 1e-12,
            1.0,
            (np.sin(N * delta / 2) / (N * np.sin(delta / 2))) ** 2
        )
        diff = np.where(np.abs(z) < 1e-12, 1.0, (np.sin(z) / z) ** 2)

    return I_0 * grating * diff + B


if __name__ == "__main__":

    data     = pd.read_csv("run_continua.txt")
    int_data = data.iloc[:, 0].to_numpy(dtype=float)

    delta_s = SPAZIO_FINE - SPAZIO_INIZIO
    v_media = delta_s / DELTA_T
    print(f"Velocità media: {v_media:.6f} m/s")

    # ── Scala spaziale ────────────────────────────────────────────────────────
    # MODIFICA 2: zero calcolato dai dati per forzare che il picco massimo
    # sia l'ordine 1 (il massimo centrale è il secondo picco, non il primo).
    # La spaziatura tra ordini adiacenti è lambda*D/d.
    N_tot   = len(int_data)
    y_raw   = np.linspace(SPAZIO_INIZIO, SPAZIO_FINE, num=N_tot)  # posizioni assolute
    Delta_y = lambda_laser * D / d  # spaziatura tra ordini ≈ 1.23 mm

    idx_max  = np.argmax(int_data)
    y_picco  = y_raw[idx_max]

    # Il picco più alto è l'ordine 1 → l'ordine 0 è una spaziatura prima
    # (segno: se il picco è a y > zero_mano, l'ordine 0 è a y più piccola)
    zero_mano    = 0.0525               # stima da righello_mano, usata come riferimento
    zero_calcolato = y_picco - Delta_y  # MODIFICA: zero forzato dal picco massimo

    print(f"Picco massimo al pixel {idx_max}, y = {y_picco*100:.3f} cm")
    print(f"Spaziatura tra ordini Delta_y = {Delta_y*1000:.3f} mm")
    print(f"zero calcolato dai dati       = {zero_calcolato*100:.4f} cm")
    print(f"zero da righello_mano         = {zero_mano*100:.4f} cm")

    zero = zero_calcolato  # usato nella I_teorica

    y_scala = y_raw - zero  # posizioni relative allo zero calcolato

    mask = y_scala > 0.03

    # ── MODIFICA 3: errore di Poisson ─────────────────────────────────────────
    sigma_poisson = np.sqrt(np.maximum(int_data, 1))  # sqrt(I), min 1 per evitare sqrt(0)

    # ── MODIFICA 4: errore sulla posizione per velocità non costante ──────────
    # Assumiamo incertezza relativa del 20% sulla costanza di v (fiducia ~80%)
    sigma_v_rel = 0.20
    sigma_v     = sigma_v_rel * v_media  # m/s

    # Il tempo di ogni punto cresce linearmente
    t         = np.linspace(0, DELTA_T, num=N_tot)
    sigma_y_i = sigma_v * t  # incertezza sulla posizione, cresce nel tempo [m]

    # ── Step 1: fit con solo errore di Poisson ────────────────────────────────
    print("\n── Step 1: fit con errore di Poisson ──")

    # Wrapper per LeastSquares (che vuole f(x, *params), senza zero_val fisso)
    def I_teorica_fit(y, I_0, N, a, B, c, m_par):
        return I_teorica(y, I_0, N, a, B, c, m_par, zero)

    ls1 = LeastSquares(y_scala, int_data, sigma_poisson, I_teorica_fit)
    ls1.mask = mask

    results = []
    for N_prova in [2, 3, 4, 5, 6]:
        m1 = Minuit(ls1, I_0=50000, N=N_prova, a=0.00013, B=0, c=0, m_par=1)
        m1.fixed["N"] = True
        m1.migrad()
        results.append((N_prova, m1.fval))
        print(f"  N={N_prova}:  chi²={m1.fval:.1f},  a={m1.values['a']:.4e},  I0={m1.values['I_0']:.1f}")

    N_migliore = min(results, key=lambda x: x[1])[0]
    print(f"\n  N migliore (step 1): {N_migliore}")

    m_step1 = Minuit(ls1, I_0=50000, N=N_migliore, a=0.00013, B=500, c=0, m_par=1)
    m_step1.fixed["N"] = True
    m_step1.limits["a"]   = (0, 0.0002)
    m_step1.limits["B"]   = (1.3e4, None)
    m_step1.migrad()

    # ── MODIFICA 5: propagazione errore posizione → errore su I ───────────────
    # dI/dy numerica calcolata sulla curva fittata dello step 1
    y_fit_vals = I_teorica_fit(y_scala, *m_step1.values)
    dI_dy      = np.gradient(y_fit_vals, y_scala)

    # Errore totale: Poisson + propagazione incertezza posizione
    sigma_tot = np.sqrt(sigma_poisson**2 + (dI_dy * sigma_y_i)**2)

    print("\n── Step 2: fit con errore totale (Poisson + posizione) ──")

    ls2 = LeastSquares(y_scala, int_data, sigma_tot, I_teorica_fit)
    ls2.mask = mask

    results2 = []
    for N_prova in [2, 3, 4, 5, 6]:
        m2tmp = Minuit(ls2, I_0=50000, N=N_prova, a=0.00013, B=0, c=0, m_par=1)
        m2tmp.fixed["N"] = True
        m2tmp.migrad()
        results2.append((N_prova, m2tmp.fval))
        print(f"  N={N_prova}:  chi²={m2tmp.fval:.1f},  a={m2tmp.values['a']:.4e},  I0={m2tmp.values['I_0']:.1f}")

    N_migliore2 = min(results2, key=lambda x: x[1])[0]
    print(f"\n  N migliore (step 2): {N_migliore2}")

    m_final = Minuit(ls2,
                     I_0=m_step1.values["I_0"],
                     N=N_migliore2,
                     a=m_step1.values["a"],
                     B=m_step1.values["B"],
                     c=m_step1.values["c"],
                     m_par=m_step1.values["m_par"])
    m_final.fixed["N"]  = True
    m_final.limits["a"] = (0, 0.0002)
    m_final.limits["B"] = (1.3e4, None)
    m_final.migrad()

    print("\n── Risultati finali ──")
    for par, val, err in zip(m_final.parameters, m_final.values, m_final.errors):
        print(f"  {par}: {val:.6e} ± {err:.6e}")

    I_0_fit  = m_final.values["I_0"]
    N_fit    = m_final.values["N"]
    a_fit    = m_final.values["a"]
    B_fit    = m_final.values["B"]
    c_fit    = m_final.values["c"]
    m_fit    = m_final.values["m_par"]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Intensità in funzione della posizione sullo schermo")
    ax.set_xlabel("Posizione (m)")
    ax.set_ylabel("Intensità (a.u.)")

    ax.plot(y_scala, int_data,
            color='blue', linewidth=1, label="Dati sperimentali")

    # MODIFICA 6: banda di errore usa sigma_tot invece del fisso 2000
    ax.fill_between(y_scala,
                    int_data - sigma_tot,
                    int_data + sigma_tot,
                    color='lightblue', alpha=0.7, label="Errore totale (Poisson + posizione)")

    ax.plot(y_scala,
            I_teorica_fit(y_scala, I_0_fit, N_fit, a_fit, B_fit, c_fit, m_fit),
            color='red', linewidth=1.5, label=f"Fit teorico (N={round(N_fit)})")

    ax.fill_between(y_scala, 0, 57000, where=mask,
                    color='mistyrose', alpha=0.4, label="Dati usati per il fit")

    # Linea verticale per mostrare dove cade lo zero calcolato
    ax.axvline(0, color='green', linestyle='--', linewidth=0.8, label=f"zero calcolato ({zero*100:.3f} cm)")

    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()