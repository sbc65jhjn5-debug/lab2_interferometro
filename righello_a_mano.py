import numpy as np

def lamb (distanza, passo, n, l, zero):
    if n == 0:
        return 0
    else:
        lam = passo * (np.sin(np.arctan(distanza / zero)) - np.sin (np.arctan(distanza / l))) / n
        return lam
    
def sigma_lambda (distanza, passo, n, l, sigma_l, zero, sigma_zero, sigma_distanza):

    if n == 0:
        return 0
    else:
        # Calcolo della derivata parziale rispetto a distanza
        partial_distanza = passo * (1 / (1 + (distanza / zero)**2) * (1 / zero) - 1 / (1 + (distanza / l)**2) * (1 / l)) / n

        # Calcolo della derivata parziale rispetto a l
        partial_l = passo * (-np.sin(np.arctan(distanza / l)) * (1 / (1 + (distanza / l)**2)) * (distanza / l**2)) / n

        # Calcolo della derivata parziale rispetto a zero
        partial_zero = passo * (np.sin(np.arctan(distanza / zero)) * (1 / (1 + (distanza / zero)**2)) * (distanza / zero**2)) / n

        # Calcolo dell'errore quadratico medio
        sigma_lam = np.sqrt((partial_distanza * sigma_distanza)**2 + (partial_l * sigma_l)**2 + (partial_zero * sigma_zero)**2)
        return sigma_lam


if __name__ == "__main__":

    d = 0.0005 # passo del righello

    misure_distanza = np.array([97.4, 97.5, 97.4, 97.5, 97.3, 97.4])
    distanza = np.mean(misure_distanza)
    sigma_distanza = np.std(misure_distanza, ddof=1)

    # distanze tra P0 e i massimi:
    distanza_primo_max = np.mean([6.0, 8.2]) * 1e-2 # -1
    distanza_secondo_max = np.mean([9.9, 11.1]) * 1e-2 # 0
    distanza_terzo_max = np.mean([11.9, 13.0]) * 1e-2 # 1
    distanza_quarto_max = np.mean([13.5, 14.5]) * 1e-2 # 2
    distanza_quinto_max = np.mean([14.9, 15.8]) * 1e-2 # 3
    print (distanza_primo_max, distanza_secondo_max, distanza_terzo_max, distanza_quarto_max, distanza_quinto_max)

    # calcolo dello 0
    zero = distanza_secondo_max / 2
    print(f"Zero: {zero:.4f} m")

    l = [distanza_primo_max - zero,
         distanza_secondo_max - zero,
         distanza_terzo_max - zero,
         distanza_quarto_max - zero,
         distanza_quinto_max - zero]
    
    n = [-1, 0, 1, 2, 3]

    lambda_vals = [lamb(distanza, d, n[i], l[i], zero) for i in range(len(n))]
    lambda_errors = [sigma_lambda(distanza, d, n[i], l[i], 0.001, zero, 0.001, sigma_distanza) for i in range(len(n))]
    for i in range(len(n)):
        print(f"Lambda for n={n[i]}: {lambda_vals[i]:.6e} m ± {lambda_errors[i]:.6e} m")