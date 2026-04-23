import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.pyplot as plt

def theta_n (r, L):
    t = np.arctan(r / L)
    return t

def sigma_theta_n (r, L, sigma_r):
    sigma_t = np.sqrt (((1 / (1 + (r/L)**2)) * sigma_r / L)**2)
    return sigma_t

def cos_theta_n (N, delta_r, lamb, d):

    # delta_r = sfasamento indotto dalle riflessioni subite dal secondo raggio
    # delta = differenza di fase
    # lambda = lunghezza d'onda del laser
    # D = differenza di cammino ottico
    # N = ordine di interferenza
    # theta = angolo tra orizzontale e raggio

    # se delta = 2 pi N allora si ha interferenza costruttiva (massimo),
    # se D = 2 d cos(theta):
    # --> delta_r lambda / (2pi) + 2 d cos(theta) = N lambda
    
    cos_theta = (N * lamb - delta_r * lamb / (2 * np.pi)) / (2 * d)
    return cos_theta


if __name__ == "__main__":

    L_mean = 1.3950 # m
    sigma_L = 0.0009

    n = np.arange (1, 12)
    r = np.array ([0.01375, 0.02075, 0.02525, 0.02925, 0.03325, 0.03625, 0.03925, 0.04175, 0.04475, 0.04675, 0.04925]) # m
    sigma_r = 0.0005

    theta = [theta_n (r_i, L_mean) for r_i in r]
    sigma_theta = [sigma_theta_n (r_i, L_mean, sigma_r) for r_i in r]
    print ("theta (gradi):", np.degrees(theta))
    print ("sigma_theta (gradi):", np.degrees(sigma_theta))

    fig, ax = plt.subplots()
    ax.errorbar (n, [np.cos(theta_i) for theta_i in theta],
                 yerr=[sigma_theta_i * np.sin(theta_i) for sigma_theta_i, theta_i in zip(sigma_theta, theta)],
                 fmt='o', 
                 capsize = 4,
                 color = 'navy',
                 label='Dati con errori'
                 )
    
    ax.set_xlabel('Ordine di interferenza (N)')
    ax.set_ylabel('cos(theta)')
    ax.set_title('Interferenza di Fabry-Perot')


    ls = LeastSquares (n,
                       [np.cos(theta_i) for theta_i in theta],
                       [sigma_theta_i * np.sin(theta_i) for sigma_theta_i, theta_i in zip(sigma_theta, theta)],
                       cos_theta_n
    )

    m = Minuit (ls, 
                delta_r = 0.0, 
                lamb = 632.8e-9, 
                d = 0.10
    )

    m.fixed["lamb"] = True
    m.migrad()

    for par, val, err in zip(m.parameters, m.values, m.errors):
        print (f"{par} = {val:.7f} ± {err:.7f}")

    delta_r_fit = m.values["delta_r"]
    d_fit = m.values["d"]

    ax.plot (n, [cos_theta_n (N_i, delta_r_fit, m.values["lamb"], d_fit) for N_i in n],
             label='Fit',
             color='crimson'
    )

    ax.legend()
    plt.show()