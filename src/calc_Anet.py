import time
import qwiic_scd4x
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import csv

def main(box_volume, leaf_area_cm2, window_size, ofname, override_temp=False,
         override_temp_c=23.0):

    DEG_2_K = 273.15

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
    with open(ofname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "co2", "temp", "rh", "vpd", "a_net"])

    try:
        while True:
            if sensor.read_measurement():
                co2 = sensor.get_co2()
                if co2 <= 0. or np.isnan(co2): # filter bad values
                    print("Invalid CO₂ reading, skipping...")
                    continue

                temp = sensor.get_temperature()
                if override_temp: # debugging - take out at some pt...
                    temp = override_temp_c
                rh = max(0, min(100, sensor.get_humidity()))
                vpd = calc_vpd(temp, rh)
                now = time.time()

                co2_window.append(co2)
                time_window.append(now)

                print(
                    f"CO₂: {co2:.1f} μmol mol⁻¹ | Temp: {temp:.1f} °C | "
                    f"RH: {rh:.1f} % | VPD: {vpd:.1f} kPa"
                )

                if len(co2_window) >= 3:
                    times = np.array(time_window)
                    co2s = np.array(co2_window)
                    slope, _ = np.polyfit(times - times[0], co2s, 1)  # ppm/s
                    temp_k = temp + DEG_2_K
                    leaf_area_m2 = leaf_area_cm2 / 10000.0
                    an_leaf = calc_anet(slope, box_volume, temp_k)
                    a_net = -an_leaf / leaf_area_m2 # umol m-2 s-1


                    print(
                        f"ΔCO₂: {slope:+.3f} μmol mol⁻¹ s⁻¹ | "
                        f"A_net: {a_net:+.2f} μmol m⁻² s⁻¹"
                    )
                    print("-" * 40)

                    with open(ofname, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([now, co2, temp, rh, vpd, a_net])
            else:
                print(".", end="", flush=True)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping measurements.")

def calc_anet(delta_ppm_s, box_volume, temp_k):
    pressure_pa = 101325.
    rgas = 8.314 # J K−1 mol−1
    volume_m3 = box_volume / 1000.0
    an_leaf = (delta_ppm_s * pressure_pa * volume_m3) / (rgas * temp_k)

    return an_leaf # umol leaf-1 s-1

def calc_vpd(temp_c, rh_percent):
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))  # kPa
    ea = es * (rh_percent / 100.0) # kPa

    return es - ea  # kPa

if __name__ == "__main__":

    box_volume = 1.2 # litres
    leaf_area_cm2 = 100.0
    window_size = 20
    ofname = "../outputs/photosynthesis_log.csv"
    main(box_volume, leaf_area_cm2, window_size, ofname, override_temp=False)
