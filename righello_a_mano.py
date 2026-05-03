import numpy as np

def lamb (distanza, passo, n, l, zero):
    if n == 0:
        return 0
    else:
        lam = passo * (np.sin(np.arctan(distanza / zero)) - np.sin (np.arctan(distanza / l))) / n
        return lam
    
def lambda_2 (distanza, passo, n, l, zero):

    if n == 0:
        return 0
    else:
        lam = passo * (np.cos (np.arctan(zero/distanza)) - np.cos (np.arctan (l/distanza))) / n
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
    
def sigma_lambda_2 (distanza, sigma_distanza, passo, n, l, sigma_l, zero, sigma_zero):

    if n == 0:
        return 0
    else:
        # Calcolo della derivata parziale rispetto a distanza
        partial_distanza = passo * (1 / ((1 + (zero / distanza)**2) * np.sqrt (1 + (zero / distanza)**2)) * (zero**2 / distanza**3) - 1 / ((1 + (l / distanza)**2) * np.sqrt(1 + (l / distanza)**2)) * (l**2 / distanza**3)) / n

        # Calcolo della derivata parziale rispetto a zero
        partial_zero = - passo * ((distanza * zero) / ((distanza**2 + zero**2) * np.sqrt(distanza**2 + zero**2))) / n

        # Calcolo della derivata parziale rispetto a l
        partial_l = passo * ((distanza * l) / ((distanza**2 + l**2) * np.sqrt(distanza**2 + l**2))) / n

        # Calcolo dell'errore quadratico medio
        sigma_lam = np.sqrt((partial_distanza * sigma_distanza)**2 + (partial_l * sigma_l)**2 + (partial_zero * sigma_zero)**2)
        return sigma_lam


if __name__ == "__main__":

    d = 0.0005 # passo del righello

    misure_distanza = np.array([97.4, 97.5, 97.4, 97.5, 97.3, 97.4])
    distanza = np.mean(misure_distanza)
    sigma_distanza = np.std(misure_distanza, ddof=1) / np.sqrt(len(misure_distanza))
    print(f"Distanza media tra righello e schermo: {distanza:.4f} cm ± {sigma_distanza:.4f} cm")

    # distanze tra P0 e i massimi:
    distanza_primo_max = np.mean([6.0, 8.2]) * 1e-2 # -1
    distanza_secondo_max = np.mean([9.9, 11.1]) * 1e-2 # 0
    distanza_terzo_max = np.mean([11.9, 13.0]) * 1e-2 # 1
    distanza_quarto_max = np.mean([13.5, 14.5]) * 1e-2 # 2
    distanza_quinto_max = np.mean([14.9, 15.8]) * 1e-2 # 3

    # sigma delle distanze tra P0 e i massimi (devono essere in metri per essere coerenti con le altre unità)
    sigma_distanza_primo_max = np.std([6.0, 8.2], ddof=1) * 1e-2 / np.sqrt(2)
    sigma_distanza_secondo_max = np.std([9.9, 11.1], ddof=1) * 1e-2 / np.sqrt(2)
    sigma_distanza_terzo_max = np.std([11.9, 13.0], ddof=1) * 1e-2 / np.sqrt(2)
    sigma_distanza_quarto_max = np.std([13.5, 14.5], ddof=1) * 1e-2 / np.sqrt(2)
    sigma_distanza_quinto_max = np.std([14.9, 15.8], ddof=1) * 1e-2 / np.sqrt(2)
    print(f"Distanza primo massimo: {distanza_primo_max:.4f} m ± {sigma_distanza_primo_max:.4f} m")
    print(f"Distanza secondo massimo: {distanza_secondo_max:.4f} m ± {sigma_distanza_secondo_max:.4f} m")
    print(f"Distanza terzo massimo: {distanza_terzo_max:.4f} m ± {sigma_distanza_terzo_max:.4f} m")
    print(f"Distanza quarto massimo: {distanza_quarto_max:.4f} m ± {sigma_distanza_quarto_max:.4f} m")
    print(f"Distanza quinto massimo: {distanza_quinto_max:.4f} m ± {sigma_distanza_quinto_max:.4f} m")

    # calcolo dello 0
    zero = distanza_secondo_max / 2
    sigma_zero = sigma_distanza_secondo_max / 2
    print(f"Zero: {zero:.4f} m ± {sigma_zero:.4f} m")

    l = [distanza_primo_max - zero,
         distanza_secondo_max - zero,
         distanza_terzo_max - zero,
         distanza_quarto_max - zero,
         distanza_quinto_max - zero]
    
    sigma_l = [sigma_distanza_primo_max + sigma_zero,
               sigma_distanza_secondo_max + sigma_zero,
               sigma_distanza_terzo_max + sigma_zero,
               sigma_distanza_quarto_max + sigma_zero,
               sigma_distanza_quinto_max + sigma_zero]
    
    n = [-1, 0, 1, 2, 3]

    lambda_vals = [lamb(distanza, d, n[i], l[i], zero) for i in range(len(n))]
    lambda_errors = [sigma_lambda(distanza, d, n[i], l[i], sigma_l[i], zero, sigma_zero, sigma_distanza) for i in range(len(n))]
    lambda_vals[1] = 6.37e-11
    lambda_errors[1] = 1e-8
    print ("\nCalcolo lambda:\n")
    for i in range(len(n)):
        print(f"Lambda for n={n[i]}: {lambda_vals[i]:.6e} m ± {lambda_errors[i]:.6e} m")

    lambda_mean = np.average (lambda_vals, weights = [1/s for s in lambda_errors])
    lambda_sigma = np.sqrt (1 / np.sum ([(1/s**2) for s in lambda_errors]))
    print ("\nValori di lambda medio:")
    print (f"{lambda_mean:.6e} ± {lambda_sigma:.6e}")

    lambda_2_vals = [lambda_2(distanza, d, n[i], l[i], zero) for i in range(len(n))]
    lambda_2_errors = [sigma_lambda_2(distanza, sigma_distanza, d, n[i], l[i], 0.001, zero, 0.001) for i in range(len(n))]
    lambda_2_vals[1] = 6.37e-11
    lambda_2_errors[1] = 3.2e-12
    print ("\nCalcolo con lambda 2:\n")
    for i in range(len(n)):
        print(f"Lambda_2 for n={n[i]}: {lambda_2_vals[i]:.6e} m ± {lambda_2_errors[i]:.6e} m")

    lambda_mean_2 = np.average (lambda_2_vals, weights = [1/s for s in lambda_2_errors])
    lambda_sigma_2 = np.sqrt (1 / np.sum ([(1/s**2) for s in lambda_2_errors]))
    print ("\nValori di lambda medio:")
    print (f"{lambda_mean_2:.6e} ± {lambda_sigma_2:.6e}")

   # abbiamo capito che l'amo qui non sa fare tanto bene le derivate...
   # ole la prossima volta tutte a mano...