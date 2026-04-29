import numpy as np
import matplotlib.pyplot as plt


def delta_d (delta_N, lamb):

    d_d = lamb * delta_N / 2
    return d_d

def delta_d_sigma (lamb, sigma_delta_N):

    sigma_d = np.sqrt((lamb /2 * sigma_delta_N)**2)
    return sigma_d


if __name__ == "__main__":

    delta_N = 51.38
    delta_N_sigma = 1.03

    lamb = 632.8e-9

    Delta_d = delta_d (delta_N, lamb)
    sigma_Delta_d = delta_d_sigma (lamb, delta_N_sigma)
    print (f"Delta d = {Delta_d*1e6:.3f} ± {sigma_Delta_d*1e6:.3f} micrometri")