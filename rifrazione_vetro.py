import numpy as np

def n (theta, delta_N, sigma_N, d, sigma_d, lamb = 632.8e-9):

    A = 1 - np.cos(theta)
    N = (2 * d - delta_N * lamb)
    D = (2 * d * A - delta_N * lamb)

    n = (N * A) / D
    sigma_n = np.sqrt((sigma_d * A * (2 * D - 2 * A * N) / D**2)**2 +
                      (sigma_N * A * (lamb * N - lamb * D) / D**2)**2)
    
    return n, sigma_n


if __name__ == "__main__":

    # Dati
    theta = np.array([np.radians(-4), np.radians(-3), np.radians(-2), np.radians(-1), np.radians(1), np.radians(2), np.radians(3), np.radians(4), np.radians(5), np.radians(10), np.radians(15)])

    delta_N = np.array([16, 9, 5, 2, 2, 4, 9, 16, 21, 41.3, 97.6])
    sigma_N = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.3, 1.4, 10.8])

    d = 0.0052 # m
    sigma_d = 0.0002 # m

    n_vals = []
    n_errs = []

    for th, N, s_N in zip (theta, delta_N, sigma_N):
        
        n_v, n_s = n (th, N, s_N, d, sigma_d)
        n_vals.append (n_v)
        n_errs.append (n_s)
        print (f"{n_v} \pm {n_s}")


    n_medio = np.average(n_vals, weights = [1/n_s for n_s in n_errs])
    sigma_n = np.std (n_vals) / np.sqrt(11)

    print (f"\nn vetro: {n_medio} \pm {sigma_n}")
