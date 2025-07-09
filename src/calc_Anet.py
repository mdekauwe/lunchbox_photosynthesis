import time
import datetime
import qwiic_scd4x
import numpy as np
from collections import deque
import csv


def main(lunchbox_volume, leaf_area_cm2, window_size, ofname,
         force_recalibrate, save_to_file=False):

    DEG_2_K = 273.15
    leaf_area_m2 = leaf_area_cm2 / 10000.0

    sensor = qwiic_scd4x.QwiicSCD4x()
    if not sensor.is_connected():
        print("Sensor not connected")
        return

    if not sensor.begin():
        print("Error while initializing sensor")
        return

    # Disable auto self-calibration
    # The SCD40 performs automatic self-calibration by default, which
    # assumes the sensor is in ambient air for at least 1 hour per day.
    # it assumes the lowest CO2 value it sees is 400 ppm and calibrates
    # accordingly...most likely, the sensor won't see "fresh air" i.e. indoors
    sensor.set_automatic_self_calibration_enabled(False)

    if force_recalibrate:
        print("\n** Waiting 3 mins for sensor to adjust in ambient air... **")
        time.sleep(300)
        print("** Performing manual calibration to 420 ppm... **")
        result = sensor.perform_forced_recalibration(420)

    print("Starting measurements...")

    co2_window = deque(maxlen=window_size)
    time_window = deque(maxlen=window_size)

    if save_to_file:
        f = open(ofname, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["time", "co2", "temp", "rh", "vpd", "a_net"])
    else:
        f = None
        writer = None

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
                    f"RH: {rh:.1f} % | VPD: {vpd:.2f} kPa"
                )

                if len(co2_window) >= window_size:
                    times = np.array(time_window)
                    co2s = np.array(co2_window)
                    slope, _ = np.polyfit(times - times[0], co2s, 1) # ppm/s
                    temp_k = temp + DEG_2_K
                    anet_leaf = calc_anet(slope, lunchbox_volume, temp_k)
                    anet_area = -anet_leaf / leaf_area_m2  # umol m-2 s-1

                    print(
                        f"ΔCO₂: {slope:+.3f} μmol mol⁻¹ s⁻¹ | "
                        f"A_net: {anet_area:+.2f} μmol m⁻² s⁻¹"
                    )
                    print("-" * 40)

                    if save_to_file:
                        writer.writerow([now_iso, f"{co2:.3f}", f"{temp:.3f}",
                                         f"{rh:.3f}", f"{vpd:.3f}",
                                         f"{anet_area:.3f}"])
                        f.flush()

                    time.sleep(6.0)
            else:
                print(".", end="", flush=True)
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping measurements.")

def calc_volume_litres(width_cm, height_cm, length_cm):

    volume_cm3 = width_cm * height_cm * length_cm
    volume_litres = volume_cm3 / 1000
    return volume_litres

def calc_anet(delta_ppm_s, lunchbox_volume, temp_k):
    # Net assimilation rate (An_leaf, umol leaf-1 s-1) calculated using the
    # ideal gas law to solve for "n" amount of substance, moles of gas
    # i.e, converts ppm s-1 into umol s-1
    #
    #            delta_CO2 × p × V
    # An_leaf = -------------------
    #                  R × T
    #
    # where:
    #   delta_CO2 = rate of CO2 change (ppm s-1)
    #   p         = pressure (Pa)
    #   V         = lunchbox_volume (m3)
    #   R         = universal gas constant (J mol⁻¹ K⁻¹)
    #   T         = temperature (K)

    pressure = 101325.  # Pa
    rgas = 8.314  # J K-1 mol-1
    volume_m3 = lunchbox_volume / 1000.0  # convert litre to m3
    an_leaf = (delta_ppm_s * pressure * volume_m3) / (rgas * temp_k)

    return an_leaf  # umol leaf s-1

def calc_vpd(temp_c, rh_percent):
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))  # kPa
    ea = es * (rh_percent / 100.0)  # kPa

    return es - ea  # kPa


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Lunchbox photosynthesis logger")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force manual recalibration to 420 ppm\
                                before starting measurements")
    parser.add_argument('--leaf_area', type=float,
                        help='Initial leaf area in cm²')
    parser.add_argument('--no_plant_pot', action='store_true',
                        help='Turn off volume correction for plant in pot')
    parser.add_argument('--save', action='store_true',
                        help='Save logged data to CSV file')
    args = parser.parse_args()

    # correct lunchbox volume for plant in pot?
    if args.no_plant_pot:
        lunchbox_volume = 1.2 # litres
    else:
        pot_volume = calc_volume_litres(5, 10, 5)
        lunchbox_volume = 1.2 - pot_volume # litres

    la = args.leaf_area if args.leaf_area and args.leaf_area > 0 else 100.0
    window_size = 12
    ofname = "../outputs/photosynthesis_log.csv"
    main(lunchbox_volume, la, window_size, ofname, args.recalibrate,
         save_to_file=args.save)
