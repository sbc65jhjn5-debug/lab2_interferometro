import numpy as np

def delta_d (delta_N, lamb):

    d_d = lamb * delta_N / 2
    return d_d

def delta_d_sigma (lamb, sigma_delta_N):

    sigma_d = np.sqrt((lamb /2 * sigma_delta_N)**2)
    return sigma_d


if __name__ == "__main__":

    delta_N = 52.44
    sigma_delta_N = 1.09

    lamb = 632.8e-9

    Delta_d = delta_d (delta_N, lamb)
    sigma_Delta_d = delta_d_sigma (lamb, sigma_delta_N)
    print (f"Delta d = {Delta_d*1e6:.3f} ± {sigma_Delta_d*1e6:.3f} micrometri")
    print (f"Fattore di correzione k = {20.0/(Delta_d*1e6):.4f}")