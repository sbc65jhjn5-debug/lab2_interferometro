from pathlib import Path

import numpy as np


def calcola_soglie(data):
    hist, edges = np.histogram(data, bins="fd")
    centers = (edges[:-1] + edges[1:]) / 2

    pesi_sx = np.cumsum(hist)
    pesi_dx = np.cumsum(hist[::-1])[::-1]

    medie_sx = np.cumsum(hist * centers) / np.maximum(pesi_sx, 1)
    medie_dx = (
        np.cumsum((hist * centers)[::-1]) / np.maximum(pesi_dx[::-1], 1)
    )[::-1]

    varianza_tra_classi = (
        pesi_sx[:-1] * pesi_dx[1:] * (medie_sx[:-1] - medie_dx[1:]) ** 2
    )
    indice_split = np.argmax(varianza_tra_classi)
    soglia_centrale = centers[indice_split]

    gruppo_scuro = data[data <= soglia_centrale]
    gruppo_chiaro = data[data > soglia_centrale]

    if gruppo_scuro.size == 0 or gruppo_chiaro.size == 0:
        raise ValueError("Impossibile separare bene i livelli chiaro/scuro")

    centro_scuro = np.median(gruppo_scuro)
    centro_chiaro = np.median(gruppo_chiaro)

    soglia_scuro = (centro_scuro + soglia_centrale) / 2
    soglia_chiaro = (centro_chiaro + soglia_centrale) / 2

    if soglia_scuro >= soglia_chiaro:
        raise ValueError("Le soglie automatiche non sono ordinate correttamente")

    return soglia_scuro, soglia_chiaro


def stato_iniziale(primo_valore, soglia_scuro, soglia_chiaro):
    if primo_valore < soglia_scuro:
        return 0
    if primo_valore >= soglia_chiaro:
        return 1

    distanza_scuro = abs(primo_valore - soglia_scuro)
    distanza_chiaro = abs(primo_valore - soglia_chiaro)
    return 0 if distanza_scuro <= distanza_chiaro else 1


path_dati = Path(__file__).with_name("data.txt")
data = np.loadtxt(path_dati)

soglia_scuro, soglia_chiaro = calcola_soglie(data)
transizioni_count = 0

# 0 = scuro, 1 = chiaro
stato = stato_iniziale(data[0], soglia_scuro, soglia_chiaro)

for val in data:
    if val < soglia_scuro and stato == 1:
        stato = 0
        transizioni_count += 1

    if val >= soglia_chiaro and stato == 0:
        stato = 1

print(f"Soglia scuro scelta automaticamente: {soglia_scuro:.0f}")
print(f"Soglia chiaro scelta automaticamente: {soglia_chiaro:.0f}")
print(f"Il numero di passaggi da chiaro a scuro è: {transizioni_count}")
