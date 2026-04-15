import numpy as np

soglia_chiaro = 2.0 * 10**4
soglia_scuro = 1.6 * 10**4
distanza_minima = 20  # numero minimo di campioni tra due oscillazioni valide


def oscillazione_valida(indice_corrente, ultimo_indice_valido, distanza_minima):
    if ultimo_indice_valido is None:
        return True
    return (indice_corrente - ultimo_indice_valido) >= distanza_minima


with open("data.txt", "r") as f:
    data = [int(line.strip()) for line in f]

chiaro_count = 0
ultimo_evento_valido = None

# 0 = scuro, 1 = chiaro
if data[0] < soglia_scuro:
    stato = 0
elif data[0] >= soglia_chiaro:
    stato = 1
else:
    raise ValueError("Il primo valore non e' ne' chiaro ne' scuro")

for i, val in enumerate(data):
    if val < soglia_scuro and stato == 1:
        if oscillazione_valida(i, ultimo_evento_valido, distanza_minima):
            stato = 0
            chiaro_count += 1
            ultimo_evento_valido = i

    if val >= soglia_chiaro and stato == 0:
        stato = 1

print(f"Il numero di passaggi validi da chiaro a scuro e': {chiaro_count}")
