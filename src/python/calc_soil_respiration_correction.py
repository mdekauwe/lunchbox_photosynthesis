#!/usr/bin/env python

import time
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from xensiv_pas_co2_sensor import CO2Sensor
from serial_port_finder import find_usb_port

def calc_soil_resp(delta_ppm_s, pressure_pa, volume_l, temp_k):
    R = 8.314  # J mol^-1 K^-1
    volume_m3 = volume_l / 1000.0  # litres to m^3
    soil_resp = (delta_ppm_s * pressure_pa * volume_m3) / (R * temp_k)
    return soil_resp

def slope_from_data(time_sec, co2_ppm, smoothing=False):
    elapsed = time_sec - time_sec[0]
    elapsed = np.round(elapsed, 2)
    elapsed -= elapsed.mean()

    if smoothing:
        from scipy.signal import medfilt, savgol_filter, butter, filtfilt

        co2_med = medfilt(co2_ppm, kernel_size=5)
        co2_smooth = savgol_filter(co2_med, window_length=19, polyorder=2)

        fs = 1 / (elapsed[1] - elapsed[0])  # sampling frequency (Hz)
        cutoff = 0.1
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        co2_filtered = filtfilt(b, a, co2_smooth)
    else:
        co2_filtered = co2_ppm

    X = sm.add_constant(elapsed)
    model = sm.OLS(co2_filtered, X)
    results = model.fit()
    slope = results.params[1]
    return slope

def calc_frustum_volume_litres(top_width_cm, base_width_cm, height_cm):
    a = top_width_cm
    b = base_width_cm
    h = height_cm
    volume_cm3 = (h / 3) * (a**2 + a*b + b**2)
    volume_litres = volume_cm3 / 1000
    return volume_litres

def calc_top_area_m2_square(width_cm, length_cm):
    return (width_cm / 100.0) * (length_cm / 100.0)

def update_plot(frame, start_time, pressure_pa, lunchbox_volume, temp_k, top_area_m2,
                measure_interval, ignore_initial_min, window_size,
                co2_window, time_window, soil_resp_values, running_avg, sensor,
                ax, line):
    current_time = time.time()
    elapsed_min = (current_time - start_time) / 60.0

    co2 = sensor.read_co2()
    co2_window.append(co2)
    time_window.append(current_time)

    soil_resp_value_m2 = None  # initialize as None every frame

    if len(co2_window) == window_size:
        time_array = np.array(time_window)
        co2_array = np.array(co2_window)

        slope_ppm_s = slope_from_data(time_array, co2_array, smoothing=True)
        soil_resp_umol_s = calc_soil_resp(slope_ppm_s, pressure_pa, lunchbox_volume, temp_k)

        if elapsed_min > ignore_initial_min:
            soil_resp_value = -soil_resp_umol_s  # uptake positive
            if soil_resp_value < 0:
                soil_resp_values.append(soil_resp_value)
                # Normalize by soil top area to get μmol/m²/s:
                soil_resp_value_m2 = soil_resp_value / top_area_m2
                running_avg.append(np.mean([v / top_area_m2 for v in soil_resp_values]))

    if running_avg:
        current_avg = running_avg[-1]
        if soil_resp_value_m2 is not None:
            print(f"Elapsed {elapsed_min:.2f} min | Soil Resp: {soil_resp_value_m2:.4f} μmol/m2/s | Mean Soil Resp: {current_avg:.4f} μmol/m²/s")
        else:
            print(f"Elapsed {elapsed_min:.2f} min | Soil Resp: --- | Mean Soil Resp: {current_avg:.4f} μmol/m²/s")
    else:
        if soil_resp_value_m2 is not None:
            print(f"Elapsed {elapsed_min:.2f} min | Soil Resp: {soil_resp_value_m2:.4f} μmol/m2/s | Mean Soil Resp: ---")
        else:
            print(f"Elapsed {elapsed_min:.2f} min | Soil Resp: --- | Mean Soil Resp: ---")

    if running_avg:
        line.set_data(range(len(running_avg)), running_avg)
        ax.set_xlim(0, len(running_avg))

    return line,

if __name__ == "__main__":

    port = find_usb_port()
    baud = 9600
    measure_interval = 1
    window_size = 41
    ignore_initial_min = 0.5

    pot_volume = calc_frustum_volume_litres(5.0, 3.4, 5.3)
    lunchbox_volume = 1.0 - pot_volume  # litres

    top_area_m2 = calc_top_area_m2_square(5, 5)

    pressure_pa = 101325.0
    temp_c = 20.0
    temp_k = temp_c + 273.15

    co2_window = deque(maxlen=window_size)
    time_window = deque(maxlen=window_size)

    soil_resp_values = []
    running_avg = []

    start_time = time.time()
    sensor = CO2Sensor(port, baud, timeout=1.0)
    sensor.reset_sensor()
    sensor.set_pressure_reference(pressure_pa)
    time.sleep(2)
    sensor.arm_sensor(rate_seconds=measure_interval)

    print("Starting measurements...")

    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_ylim(-3, 0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Window number")
    ax.set_ylabel("Mean Soil Respiration (μmol/m²/s)")


    ani = animation.FuncAnimation(
        fig, update_plot, interval=1000, cache_frame_data=False,
        fargs=(start_time, pressure_pa, lunchbox_volume, temp_k, top_area_m2,
               measure_interval, ignore_initial_min, window_size,
               co2_window, time_window, soil_resp_values, running_avg, sensor,
               ax, line))

    plt.show()

    try:
        plt.pause(3600*24)
    except KeyboardInterrupt:
        print("\nStopping measurements.")

        negative_soil_resp = np.array([val for val in soil_resp_values if val < 0])

        if len(negative_soil_resp) > 0:
            mean = np.mean(negative_soil_resp)
            std = np.std(negative_soil_resp)
            filtered = negative_soil_resp[negative_soil_resp > (mean - 3*std)]
            soil_resp_correction_umol_s = np.mean(filtered)
            soil_resp_correction_umol_m2_s = soil_resp_correction_umol_s / top_area_m2
            print(f"Estimated soil respiration correction : {soil_resp_correction_umol_m2_s:.4f} umol/m2/s")
        else:
            print("No negative soil respiration values found to estimate soil respiration.")

    finally:
            sensor.close()
