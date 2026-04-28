import numpy as np
import matplotlib.pyplot as plt


def delta_d (theta, delta_N, lamb):

    cos_theta = np.cos(theta)
    d = lamb * delta_N / (2 * cos_theta)
    return d

def delta_d_sigma (theta, delta_N, lamb, sigma_theta, sigma_delta_N):

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    d = delta_d(theta, delta_N, lamb)

    sigma_d = np.sqrt( (lamb * delta_N * sin_theta / (2 * cos_theta**2) * sigma_theta)**2 + 
                       (lamb / (2 * cos_theta) * sigma_delta_N)**2)
    return sigma_d

if __name__ == "__main__":

    delta_N = 51.38
    delta_N_sigma = 1.03

    lamb = 632.8e-9

    # Stimiamo theta perché siamo scemi e ci siamo dimenticati di misurarlo...

    d_finto = 20e-6 # m
    sigma_d_finto = 1e-6 # m

    theta_stimato = np.arccos (lamb * delta_N / (2 * d_finto))
    sigma_theta_stimato = np.sqrt ((1/np.sqrt(1 - (lamb * delta_N / (2 * d_finto))**2) * lamb /(2*d_finto) * delta_N_sigma)**2 +
                                   (1/np.sqrt(1 - (lamb * delta_N / (2 * d_finto))**2) * lamb * delta_N / (2 * d_finto**2) * sigma_d_finto)**2)

    print (f"theta stimato = {np.degrees(theta_stimato):.3f} ± {np.degrees(sigma_theta_stimato):.3f} gradi")

    # Stimiamo "d_vero" con theta stimato e delta_N misurato

    d_vero = delta_d(theta_stimato, delta_N, lamb)
    sigma_d_vero = delta_d_sigma(theta_stimato, delta_N, lamb, sigma_theta_stimato, delta_N_sigma)
    print (f"d vero = {d_vero*1e6:.3f} ± {sigma_d_vero*1e6:.3f} micrometri")