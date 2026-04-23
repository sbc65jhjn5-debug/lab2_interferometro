import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(rf"run_continua.txt")
y_data = data.iloc[:, 0]
x_data = list (range(1, len(y_data) + 1))

fig, ax = plt.subplots()

ax.set_title("LDR data")
ax.set_xlabel("Time ($10^{-1}$ s)")
ax.set_ylabel("LDR value")

ax.plot(x_data, y_data, label="LDR", marker = '', color='blue', linewidth=1)

plt.legend ()
plt.grid (True)
plt.show ()