# run continua svolta in
# tempo: 22.4 secondi
# spazio: da 6.0 a 14.0 centimetri
#
# Per questi dati e meglio fittare direttamente la periodicita delle frange,
# aggiungendo anche un inviluppo lento:
# I(x) = offset + slope * (x - x_c)
#      + amp * [1 + env_amp * sinc^2((x - env_x0) / env_width)] * cos^2(k * (x - x0))
#
# Il fit precedente lasciava liberi d, lambda e L, ma dai dati compare solo
# il rapporto d / (lambda * L). Inoltre le guess iniziali portavano facilmente
# il coseno in regime quasi parabolico.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.signal import find_peaks, savgol_filter


DELTA_T = 22.4
SPAZIO_INIZIO_CM = 6.0
SPAZIO_FINE_CM = 14.0

# Se vuoi ricavare d dal fit, imposta qui i valori noti del setup.
L_CM = 100.0
LAMBDA_CM = 632.8e-7  # 632.8 nm = 6.328e-5 cm


def theoretical_intensity(x, offset, slope, amp, k, x0, env_amp, env_width, env_x0):
    x_center = np.mean(x)
    baseline = offset + slope * (x - x_center)
    envelope = 1.0 + env_amp * np.sinc((x - env_x0) / env_width) ** 2
    return baseline + amp * envelope * np.cos(k * (x - x0)) ** 2


def estimate_initial_parameters(x_data, y_data):
    y_smooth = savgol_filter(y_data, 21, 3)
    prominence = max(1500.0, 0.08 * (np.max(y_data) - np.min(y_data)))
    distance = max(8, len(y_data) // 20)

    peaks, _ = find_peaks(y_smooth, prominence=prominence, distance=distance)

    if len(peaks) >= 2:
        fringe_period = float(np.median(np.diff(x_data[peaks])))
    else:
        fringe_period = max((x_data[-1] - x_data[0]) / 5.0, 1e-3)

    offset0 = float(np.percentile(y_data, 20))
    amp0 = float(0.5 * (np.percentile(y_data, 90) - np.percentile(y_data, 10)))
    slope0 = float((y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]))
    k0 = float(np.pi / fringe_period)
    x0 = float(x_data[np.argmax(y_smooth)])

    env_amp0 = 1.0
    env_width0 = 1.5
    env_x00 = x0

    return peaks, y_smooth, offset0, slope0, amp0, k0, x0, env_amp0, env_width0, env_x00


if __name__ == "__main__":
    data = pd.read_csv("run_continua.txt")
    y_data = data.iloc[:, 0].to_numpy(dtype=float)

    delta_s = SPAZIO_FINE_CM - SPAZIO_INIZIO_CM
    v_media = delta_s / DELTA_T
    print(f"Velocita media: {v_media:.4f} cm/s")

    spazio_axis_assoluto = np.linspace(SPAZIO_INIZIO_CM, SPAZIO_FINE_CM, len(y_data))
    (
        peaks,
        y_smooth,
        offset0,
        slope0,
        amp0,
        k0,
        x0,
        env_amp0,
        env_width0,
        env_x00,
    ) = estimate_initial_parameters(spazio_axis_assoluto, y_data)

    if len(peaks) >= 2:
        x_zero = float(spazio_axis_assoluto[peaks[1]])
        zero_label = "secondo massimo grande"
    elif len(peaks) == 1:
        x_zero = float(spazio_axis_assoluto[peaks[0]])
        zero_label = "primo massimo trovato"
    else:
        x_zero = float(spazio_axis_assoluto[np.argmax(y_smooth)])
        zero_label = "massimo globale smussato"

    spazio_axis = spazio_axis_assoluto - x_zero
    x0 -= x_zero
    env_x00 -= x_zero

    print(f"Picchi trovati: {len(peaks)}")
    if len(peaks) >= 2:
        period0 = np.median(np.diff(spazio_axis[peaks]))
        print(f"Periodo iniziale stimato: {period0:.4f} cm")
    print(f"Zero dello spazio fissato a {zero_label}: x = {x_zero:.4f} cm")
    print(
        f"Guess iniziali: offset={offset0:.1f}, slope={slope0:.1f}, "
        f"amp={amp0:.1f}, k={k0:.4f}, x0={x0:.4f}, "
        f"env_amp={env_amp0:.2f}, env_width={env_width0:.2f}, env_x0={env_x00:.4f}"
    )

    sigma = np.full_like(y_data, max(np.std(y_data) * 0.12, 1.0))
    ls = LeastSquares(spazio_axis, y_data, sigma, theoretical_intensity)

    m = Minuit(
        ls,
        offset=offset0,
        slope=slope0,
        amp=amp0,
        k=k0,
        x0=x0,
        env_amp=env_amp0,
        env_width=env_width0,
        env_x0=env_x00,
    )
    m.limits["offset"] = (0.0, None)
    m.limits["amp"] = (0.0, None)
    m.limits["k"] = (0.0, None)
    m.limits["env_amp"] = (0.0, 10.0)
    m.limits["env_width"] = (0.2, 20.0)
    m.limits["env_x0"] = (spazio_axis.min() - 5.0, spazio_axis.max() + 5.0)
    m.migrad()
    m.hesse()

    offset_fit = m.values["offset"]
    slope_fit = m.values["slope"]
    amp_fit = m.values["amp"]
    k_fit = m.values["k"]
    x0_fit = m.values["x0"]
    env_amp_fit = m.values["env_amp"]
    env_width_fit = m.values["env_width"]
    env_x0_fit = m.values["env_x0"]

    fringe_period_fit = np.pi / k_fit
    d_fit_cm = k_fit * LAMBDA_CM * L_CM / np.pi
    rmse_fit = np.sqrt(
        np.mean(
            (
                y_data
                - theoretical_intensity(
                    spazio_axis,
                    offset_fit,
                    slope_fit,
                    amp_fit,
                    k_fit,
                    x0_fit,
                    env_amp_fit,
                    env_width_fit,
                    env_x0_fit,
                )
            )
            ** 2
        )
    )

    print(
        f"Fit: offset={offset_fit:.1f}, slope={slope_fit:.1f}, "
        f"amp={amp_fit:.1f}, k={k_fit:.4f}, x0={x0_fit:.4f}, "
        f"env_amp={env_amp_fit:.3f}, env_width={env_width_fit:.3f}, env_x0={env_x0_fit:.4f}"
    )
    print(f"Periodo fit: {fringe_period_fit:.4f} cm")
    print(f"RMSE fit: {rmse_fit:.1f}")
    print(
        f"d ricavato con lambda={LAMBDA_CM:.4e} cm e L={L_CM:.1f} cm: "
        f"{d_fit_cm:.4e} cm"
    )

    x_fit = np.linspace(spazio_axis.min(), spazio_axis.max(), 1000)
    y_fit = theoretical_intensity(
        x_fit,
        offset_fit,
        slope_fit,
        amp_fit,
        k_fit,
        x0_fit,
        env_amp_fit,
        env_width_fit,
        env_x0_fit,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("LDR data with fit")
    ax.set_xlabel("Space (cm)")
    ax.set_ylabel("LDR value")
    ax.plot(spazio_axis, y_data, label="Data", color="tab:blue", lw=1.5)
    ax.plot(spazio_axis, y_smooth, label="Smoothed data", color="tab:orange", lw=2)
    if len(peaks) > 0:
        ax.scatter(
            spazio_axis[peaks],
            y_smooth[peaks],
            label="Peaks",
            color="tab:red",
            zorder=3,
        )
    ax.plot(x_fit, y_fit, label="Fit", color="purple", lw=2.5)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
