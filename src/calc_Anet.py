import time
import datetime
import qwiic_scd4x
import numpy as np
from collections import deque
import csv


def main(box_volume, leaf_area_cm2, window_size, ofname):

    DEG_2_K = 273.15

    leaf_area_m2 = leaf_area_cm2 / 10000.0

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
                    if co2 <= 0. or np.isnan(co2):
                        print("Invalid CO₂ reading, skipping...")
                        continue

                    temp = sensor.get_temperature()
                    if temp < -40 or temp > 60:
                        print("Temperature out of range, skipping...")
                        continue

                    rh = max(0, min(100, sensor.get_humidity()))
                    vpd = calc_vpd(temp, rh)
                    now_iso = datetime.datetime.now().isoformat()

                    co2_window.append(co2)
                    time_window.append(time.time())

                    print(
                        f"CO₂: {co2:.1f} μmol mol⁻¹ | Temp: {temp:.1f} °C | "
                        f"RH: {rh:.1f} % | VPD: {vpd:.1f} kPa"
                    )

                    if len(co2_window) >= window_size:
                        times = np.array(time_window)
                        co2s = np.array(co2_window)
                        slope, _ = np.polyfit(times - times[0], co2s, 1)  # ppm/s
                        temp_k = temp + DEG_2_K
                        anet_leaf = calc_anet(slope, box_volume, temp_k)
                        anet_area = -anet_leaf / leaf_area_m2  # µmol m⁻² s⁻¹

                        print(
                            f"ΔCO₂: {slope:+.3f} μmol mol⁻¹ s⁻¹ | "
                            f"A_net: {anet_area:+.2f} μmol m⁻² s⁻¹"
                        )
                        print("-" * 40)

                        writer.writerow([now_iso, f"{co2:.3f}", f"{temp:.3f}",
                                         f"{rh:.3f}", f"{vpd:.3f}",
                                         f"{anet_area:.3f}"])
                        f.flush()
                else:
                    print(".", end="", flush=True)

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nStopping measurements.")

def calc_anet(delta_ppm_s, box_volume, temp_k):
    pressure = 101325. # Pa
    rgas = 8.314  # J K⁻¹ mol⁻¹
    volume_m3 = box_volume / 1000.0 # convert litre to m³
    an_leaf = (delta_ppm_s * pressure * volume_m3) / (rgas * temp_k)

    return an_leaf # µmol leaf s⁻¹

def calc_vpd(temp_c, rh_percent):
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))  # kPa
    ea = es * (rh_percent / 100.0) # kPa

    return es - ea # kPa


if __name__ == "__main__":

    box_volume = 1.2  # litres
    leaf_area_cm2 = 100.0
    window_size = 20
    ofname = "../outputs/photosynthesis_log.csv"
    main(box_volume, leaf_area_cm2, window_size, ofname)
