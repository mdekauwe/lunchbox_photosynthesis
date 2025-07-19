#!/usr/bin/env python

import csv
from datetime import datetime, UTC
import matplotlib.pyplot as plt


def main(fname, lunchbox_volume):

    timestamps, co2_ppm = read_co2_csv(fname)

    delta_ppm_s = []
    times = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]  # seconds
        dppm = co2_ppm[i] - co2_ppm[i-1]
        delta_ppm_s.append(dppm / dt)
        times.append(timestamps[i])

    anet = [calc_anet(dppm_s, lunchbox_volume) for dppm_s in delta_ppm_s]

    # Convert unix timestamps to datetime for plotting
    times_dt = [datetime.fromtimestamp(t, UTC) for t in times]

    plt.figure(figsize=(10,6))
    plt.plot(times_dt, anet, label=r'$A_{net}$ ($\mu$mol s$^{-1}$)')
    plt.ylabel(r'Net Assimilation Rate ($\mu$mol s$^{-1}$)')
    plt.xlabel('Time (UTC)')
    plt.tight_layout()
    plt.show()

def calc_volume_litres(width_cm, height_cm, length_cm):
    volume_cm3 = width_cm * height_cm * length_cm
    volume_litres = volume_cm3 / 1000

    return volume_litres

def calc_anet(delta_ppm_s, lunchbox_volume, temp_k=295.15):
    pressure = 101325.  # Pa
    rgas = 8.314  # J K-1 mol-1
    volume_m3 = lunchbox_volume / 1000.0  # litre to m3
    an_leaf = (delta_ppm_s * pressure * volume_m3) / (rgas * temp_k)

    return an_leaf  # umol leaf s-1

def read_co2_csv(filename):
    timestamps = []
    co2_ppm = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)

        # Skip all comment lines starting with #
        for row in reader:
            if not row:
                continue
            if row[0].startswith('#'):
                continue
            else:
                # This is the header row; skip it
                break

        # Now read all remaining rows as data
        for row in reader:
            if not row:
                continue
            unix_ts = int(row[0])
            co2 = float(row[2])
            timestamps.append(unix_ts)
            co2_ppm.append(co2)

    return timestamps, co2_ppm


if __name__ == "__main__":

    fname = "/Users/xj21307/Desktop/PAS_CO2_datalog_20250718-162922.csv"
    lunchbox_volume = calc_volume_litres(width_cm=17.5, height_cm=5, length_cm=12)  # 1L
    print(lunchbox_volume)
    main(fname, lunchbox_volume)
