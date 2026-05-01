import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(rf"run2_pompa.txt")
y_data = data.iloc[:, 0]
x_data = list (range(1, len(y_data) + 1))

fig, ax = plt.subplots()

ax.set_title("Valori LDR durante il funzionamento della pompa")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("LDR value")

ax.axvspan(630, 760, color='mistyrose', alpha=0.7, label="Area scartata", zorder=0)
ax.plot(x_data, y_data, label="LDR", marker = '', color='blue', linewidth=1)

plt.legend (loc='upper right')
plt.grid (True)
plt.show ()
