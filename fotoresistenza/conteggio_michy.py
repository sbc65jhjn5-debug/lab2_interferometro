import numpy as np

# per run 1, 2, 3, 4, 5, 6:
#soglia_chiaro = 1.2 * 10**4
#soglia_scuro = 1.0 * 10**4

soglia_chiaro = 3.25 * 10**4
soglia_scuro = 2.5 * 10**4


with open ("data.txt", "r") as f:
    data = [int(line.strip()) for line in f]

chiaro_count = 0
# 0 = scuro, 1 = chiaro

if data[0] < soglia_scuro:
    stato = 0
elif data[0] >= soglia_chiaro:
    stato = 1
else:
    raise ValueError ("Il primo valore non è né chiaro né scuro")

for val in data:

    if val < soglia_scuro and stato == 1:
        stato = 0
        chiaro_count += 1

    if val >= soglia_chiaro and stato == 0:
        stato = 1

print (f"Il numero di passaggi da scuro a chiaro è: {chiaro_count}")