import time
import qwiic_scd4x
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

CHAMBER_VOLUME_LITERS = 1.2
LEAF_AREA_CM2 = 100.0
WINDOW_SIZE = 6

PRESSURE = 101325  # Pa
R = 8.314  # J mol K-1

# Store plotting history
PLOT_WINDOW = 300  # seconds of Anet data to keep

def ppm_to_umol_s(delta_ppm_s, volume_liters, temp_K, pressure_pa):
    volume_m3 = volume_liters / 1000.0
    mol_flux = (delta_ppm_s * pressure_pa * volume_m3) / (R * temp_K)
    return mol_flux

def setup_plot():
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'g-')
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Net Photosynthesis (μmol m$^{-2}$ s$^{-1}$)")
    ax.grid(True)
    return fig, ax, line

def update_plot(line, ax, times, anets):
    if len(times) == 0:
        return
    times_rel = [(t - times[0]) / 60 for t in times]  # minutes
    line.set_xdata(times_rel)
    line.set_ydata(anets)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

def main():
    sensor = qwiic_scd4x.QwiicSCD4x()

    if not sensor.is_connected():
        print("Sensor not connected")
        return

    if not sensor.begin():
        print("Error while initializing sensor")
        return

    co2_window = deque(maxlen=WINDOW_SIZE)
    time_window = deque(maxlen=WINDOW_SIZE)

    # For plotting
    anet_times = deque()
    anet_values = deque()

    fig, ax, line = setup_plot()

    print("Starting measurements...")

    try:
        while True:
            if sensor.read_measurement():
                co2 = sensor.get_co2()
                temp = sensor.get_temperature()
                rh = sensor.get_humidity()
                now = time.time()

                co2_window.append(co2)
                time_window.append(now)

                print(f"CO2: {co2:.1f} ppm | Temp: {temp:.1f} degC | RH: {rh:.1f}%")

                if len(co2_window) >= 3:
                    times = np.array(time_window)
                    co2s = np.array(co2_window)
                    slope, _ = np.polyfit(times - times[0], co2s, 1)  # ppm/s

                    temp_K = temp + 273.15
                    mol_flux = ppm_to_umol_s(slope, CHAMBER_VOLUME_LITERS, temp_K, PRESSURE)
                    leaf_area_m2 = LEAF_AREA_CM2 / 10000.0
                    A_net = -mol_flux / leaf_area_m2

                    print(f" delta_CO2: {slope:+.3f} ppm/s | Anet: {A_net:+.2f} μmol m-2 s-1")
                    print("-" * 40)

                    # Store and update plot
                    anet_times.append(now)
                    anet_values.append(A_net)

                    # Keep only data within plot window
                    while anet_times and (now - anet_times[0]) > PLOT_WINDOW:
                        anet_times.popleft()
                        anet_values.popleft()

                    update_plot(line, ax, list(anet_times), list(anet_values))

            else:
                print(".", end="")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping measurements.")

if __name__ == "__main__":
    main()
