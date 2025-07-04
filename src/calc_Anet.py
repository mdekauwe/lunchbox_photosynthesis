import time
import qwiic_scd4x
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def ppm_to_umol_s(delta_ppm_s, box_volume, temp_k,
                  pressure_pa, RGAS=8.314):
    volume_m3 = box_volume / 1000.0
    mol_flux = (delta_ppm_s * pressure_pa * volume_m3) / (RGAS * temp_k)
    return mol_flux


def main(box_volume, leaf_area_cm2, window_size):
    pressure_pa = 101325

    sensor = qwiic_scd4x.QwiicSCD4x()
    if not sensor.is_connected():
        print("Sensor not connected")
        return

    if not sensor.begin():
        print("Error while initializing sensor")
        return

    co2_window = deque(maxlen=window_size)
    time_window = deque(maxlen=window_size)

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

                print(
                    f"CO₂: {co2:.1f} ppm | Temp: {temp:.1f} °C | "
                    f"RH: {rh:.1f}%"
                )

                if len(co2_window) >= 3:
                    times = np.array(time_window)
                    co2s = np.array(co2_window)
                    slope, _ = np.polyfit(times - times[0], co2s, 1)  # ppm/s

                    print(times)
                    print(co2s)
                    import sys; sys.exit()
                    temp_k = temp + 273.15
                    leaf_area_m2 = leaf_area_cm2 / 10000.0

                    mol_flux = ppm_to_umol_s(slope, box_volume,
                                             temp_k, pressure_pa)

                    a_net = -mol_flux / leaf_area_m2

                    print(
                        f"ΔCO₂: {slope:+.3f} ppm/s | "
                        f"A_net: {a_net:+.2f} μmol m⁻² s⁻¹"
                    )
                    print("-" * 40)

            else:
                print(".", end="", flush=True)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping measurements.")


if __name__ == "__main__":
    box_volume = 1.2
    leaf_area_cm2 = 100.0
    window_size = 6

    main(box_volume, leaf_area_cm2, window_size)
