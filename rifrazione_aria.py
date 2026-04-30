import numpy as np

def m (delta_P, d, sigma_d, delta_N, sigma_N, lamb=632.8e-9):
    m = delta_N * lamb / (2 * d * delta_P)
    sigma_m = np.sqrt((sigma_N * lamb / (2 * d * delta_P))**2 + (delta_N * lamb / (2 * d**2 * delta_P) * sigma_d)**2)

    return m, sigma_m

def n (m, sigma_m, delta_P):
    n = 1 + m * delta_P
    sigma_n = sigma_m * delta_P
    return n, sigma_n

if __name__ == "__main__":

    delta_P = 40e3 # Pa
    d = 3.04e-2 # m
    sigma_d = 0.05e-2 # m

    delta_N = np.mean([10,9,9,7,8,9,11,10])
    sigma_N = np.std([10,9,9,7,8,9,11,10])/np.sqrt(8)

    print (f"delta N = {delta_N} +/- {sigma_N} frange")

    m_val, sigma_m = m (delta_P, d, sigma_d, delta_N, sigma_N)
    print (f"m = {m_val * 10e9:.3f} +/- {sigma_m * 10e9:.3f} 10^9 Pa^-1")

    n_val, sigma_n = n (m_val, sigma_m, delta_P)
    print (f"n = {n_val:.6f} +/- {sigma_n:.6f}")