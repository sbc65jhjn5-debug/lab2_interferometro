import numpy as np

soglia_chiaro = 1.6 * 10**4
soglia_scuro = 6.7 * 10**3


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