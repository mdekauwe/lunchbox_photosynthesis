import time
import qwiic_scd4x
import numpy as np
from collections import deque

CHAMBER_VOLUME_LITERS = 1.2
LEAF_AREA_CM2 = 100.0
WINDOW_SIZE = 6

SEA_LEVEL_PRESSURE_PA = 101325
R = 8.314  # J/mol/K

def ppm_to_umol_s(delta_ppm_s, volume_liters, temp_K, pressure_pa):
    volume_m3 = volume_liters / 1000.0
    mol_flux = (delta_ppm_s * pressure_pa * volume_m3) / (R * temp_K)
    return mol_flux

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

    print("Starting measurements...")

    try:
        while True:
            if sensor.read_measurement():  # crucial: only read values if fresh data available
                co2 = sensor.get_co2()
                temp = sensor.get_temperature()
                rh = sensor.get_humidity()
                now = time.time()

                co2_window.append(co2)
                time_window.append(now)

                print(f"CO2: {co2:.1f} ppm | Temp: {temp:.1f} °C | RH: {rh:.1f}%")

                if len(co2_window) >= 3:
                    times = np.array(time_window)
                    co2s = np.array(co2_window)
                    slope, _ = np.polyfit(times - times[0], co2s, 1)  # ppm/s

                    temp_K = temp + 273.15
                    mol_flux = ppm_to_umol_s(slope, CHAMBER_VOLUME_LITERS, temp_K, SEA_LEVEL_PRESSURE_PA)
                    leaf_area_m2 = LEAF_AREA_CM2 / 10000.0
                    A_net = -mol_flux / leaf_area_m2

                    print(f"→ ΔCO2: {slope:+.3f} ppm/s | Anet: {A_net:+.2f} μmol m⁻² s⁻¹")
                    print("-" * 40)
            else:
                print(".", end="")  # waiting for new data

            time.sleep(0.5)  # poll frequently

    except KeyboardInterrupt:
        print("\nStopping measurements.")

if __name__ == "__main__":
    main()

