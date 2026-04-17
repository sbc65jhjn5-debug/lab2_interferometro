import numpy as np

if __name__ == "__main__":

    d = 0.0005 # passo del righello

    misure_distanza = np.array([97.4, 97.5, 97.4, 97.5, 97.3, 97.4])
    distanza = np.mean(misure_distanza)
    sigma_distanza = np.std(misure_distanza, ddof=1)

    # distanze tra P0 e i massimi:
    distanza_primo_max = np.mean([6.0, 8.2]) # -1
    distanza_secondo_max = np.mean([9.9, 11.1]) # 0
    distanza_terzo_max = np.mean([11.9, 13.0]) # 1
    distanza_quarto_max = np.mean([13.5, 14.5]) # 2
    distanza_quinto_max = np.mean([14.9, 15.8]) # 3