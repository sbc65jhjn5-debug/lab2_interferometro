import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.pyplot as plt
from scipy.stats import chi2


def theta_n (r, L):
    t = np.arctan(r / L)
    return t


def sigma_theta_n (r, L, sigma_r):
    sigma_t = np.sqrt (((1 / (1 + (r/L)**2)) * sigma_r / L)**2)
    return sigma_t


def cos_theta_n (p, a, lamb, d):

    # Se indichiamo con p = 0, 1, 2, ... il numero progressivo delle frange
    # andando dal centro verso l'esterno, l'ordine assoluto non e noto:
    # N_p = N_0 - p.
    #
    # Quindi sui dati osservabili il modello corretto e lineare:
    # 2 d cos(theta_p) / lambda + delta_r / (2 pi) = N_0 - p
    # --> cos(theta_p) = a - p * lambda / (2 d)
    #
    # Il parametro a assorbe il termine costante che dipende da N_0 e delta_r.
    cos_theta = a - p * lamb / (2 * d)
    return cos_theta


if __name__ == "__main__":

    L_mean = 1.3950 # m
    sigma_L = 0.0009

    p = np.arange (11)
    r = np.array ([0.01375, 0.02075, 0.02525, 0.02925, 0.03325, 0.03625, 0.03925, 0.04175, 0.04475, 0.04675, 0.04925]) # m
    sigma_r = 0.0005

    theta = [theta_n (r_i, L_mean) for r_i in r]
    sigma_theta = [sigma_theta_n (r_i, L_mean, sigma_r) for r_i in r]
    print ("theta (gradi):", np.degrees(theta))
    print ("sigma_theta (gradi):", np.degrees(sigma_theta))

    fig, ax = plt.subplots()
    ax.errorbar (p, [np.cos(theta_i) for theta_i in theta],
                 yerr=[sigma_theta_i * np.sin(theta_i) for sigma_theta_i, theta_i in zip(sigma_theta, theta)],
                 fmt='o', 
                 capsize = 4,
                 color = 'navy',
                 label='Dati con errori'
                 )
    
    ax.set_xlabel('Numero progressivo della frangia $N$')
    ax.set_ylabel('$\\cos (\\theta)$')
    ax.set_title('Interferenza di Fabry-Perot')


    ls = LeastSquares (p,
                       [np.cos(theta_i) for theta_i in theta],
                       [sigma_theta_i * np.sin(theta_i) for sigma_theta_i, theta_i in zip(sigma_theta, theta)],
                       cos_theta_n
    )

    m = Minuit (ls, 
                a = np.cos(theta[0]),
                lamb = 632.8e-9, 
                d = 0.005
    )

    m.fixed["lamb"] = True
    m.limits["a"] = (0.0, 1.1)
    m.limits["d"] = (1e-5, None)
    m.migrad()
    m.hesse()

    for par, val, err in zip(m.parameters, m.values, m.errors):
        print (f"{par} = {val:.7f} ± {err:.7f}")

    a_fit = m.values["a"]
    d_fit = m.values["d"]

    # Chi quadro
    chi2_val = m.fval
    p_value = chi2.sf (chi2_val, m.ndof)
    print (f"Chi quadro: {chi2_val:.2f},\ngradi di libertà: {m.ndof},\np-value: {p_value:.3f}")

    # Grafico del fit sopra i punti

    ax.plot (p, [cos_theta_n (p_i, a_fit, m.values["lamb"], d_fit) for p_i in p],
             label=f'$y = {a_fit:.5f} - (\\lambda  / (2 * {d_fit:.5f})) \\; x$',
             color='crimson'
    )

    print (f"d = {d_fit * 1e3:.5f} ± {m.errors['d'] * 1e3:.5f} mm")

    plt.grid (True)
    plt.legend()
    plt.show()