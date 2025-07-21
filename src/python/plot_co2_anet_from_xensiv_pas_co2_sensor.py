#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import argparse
import numpy as np
import serial

from xensiv_pas_co2_sensor import CO2Sensor


class LunchboxLogger:
    def __init__(self, port, baud, lunchbox_volume, temp_c, leaf_area_cm2,
                 window_size, measure_interval=10, timeout=1.0,
                 plot_duration_min=10):

        self.temp_k = temp_c + 273.15
        self.pressure = 101325.  # Pa
        self.leaf_area_m2 = leaf_area_cm2 / 10000.0
        self.lunchbox_volume = lunchbox_volume
        self.window_size = window_size
        self.measure_interval = measure_interval
        self.plot_duration_min = plot_duration_min
        self.plot_duration_s = plot_duration_min * 60
        self.interval_ms = 60
        self.max_len = int(self.plot_duration_s / (self.interval_ms / 1000))

        # Data buffers
        self.xs = deque(maxlen=self.max_len)
        self.ys_co2 = deque(maxlen=self.max_len)
        self.ys_anet = deque(maxlen=self.max_len)
        self.co2_window = deque(maxlen=window_size)
        self.time_window = deque(maxlen=window_size)

        # Setup sensor
        self.sensor = CO2Sensor(port, baud, timeout)
        try:
            self.sensor.arm_sensor()
        except Exception as e:
            print(f"Failed to arm sensor: {e}")
            self.sensor.close()
            raise

        # Plot setup
        self.fig, self.ax_anet = plt.subplots(figsize=(12, 6))
        self.ax_co2 = self.ax_anet.twinx()
        self._setup_axes()

        self.line_anet, = self.ax_anet.plot([], [], lw=2, color="royalblue", \
                            label="Anet")
        self.line_co2, = self.ax_co2.plot([], [], lw=2, color="seagreen", \
                            label="CO₂")

        self.start_time = time.time()
        self.last_measure_time = 0

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, blit=False,
                                      interval=self.interval_ms,
                                      cache_frame_data=False,)
        plt.tight_layout()
        plt.show()
        self.sensor.close()

    def _setup_axes(self):
        self.ax_anet.set_xlabel("Elapsed Time (min)")
        self.ax_anet.set_ylabel("Net assimilation rate (μmol m⁻² s⁻¹)",
                                color="black")
        self.ax_anet.set_ylim(-1, 5)
        self.ax_anet.tick_params(axis="y", labelcolor="black")
        self.ax_anet.set_xlim(0, self.plot_duration_min)

        self.ax_co2.set_ylabel("CO₂ (ppm)", color="black")
        self.ax_co2.set_ylim(0, 1000)
        self.ax_co2.tick_params(axis="y", labelcolor="black")

    def calc_anet(self, delta_ppm_s):
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
        rgas = 8.314  # J K-1 mol-1
        volume_m3 = self.lunchbox_volume / 1000.0  # litre to m3
        an = (delta_ppm_s * self.pressure * volume_m3) / (rgas * self.temp_k)

        return an # umol leaf s-1

    def update(self, frame):
        current_time = time.time()
        elapsed_s = current_time - self.start_time
        elapsed_min = elapsed_s / 60

        if (current_time - self.last_measure_time) >= self.measure_interval:
            try:
                co2 = self.sensor.read_co2()
                self.last_measure_time = current_time
            except Exception as e:
                print(f"Read error: {e}")
                return self.line_co2, self.line_anet

            self.co2_window.append(co2)
            self.time_window.append(current_time)

            if len(self.co2_window) >= self.window_size:
                co2_array = np.array(self.co2_window)
                time_array = np.array(self.time_window)
                elapsed = time_array - time_array[0]
                slope, _ = np.polyfit(elapsed, co2_array, 1)
                anet_leaf = self.calc_anet(slope)
                anet_area = -anet_leaf / self.leaf_area_m2

                print(f"Time: {elapsed_min:.2f} min | "
                      f"CO₂: {co2:.3f} ppm | "
                      f"A_net: {anet_area:+.2f} μmol m⁻² s⁻¹")

                if elapsed_s > self.plot_duration_s:
                    shifted_xs = [x - (elapsed_min - \
                                  self.plot_duration_min) for x in self.xs]
                    self.xs.clear()
                    self.xs.extend(shifted_xs)
                    self.xs.append(self.plot_duration_min)
                else:
                    self.xs.append(elapsed_min)

                self.ys_co2.append(co2)
                self.ys_anet.append(anet_area)

                co2_min = min(self.ys_co2) if self.ys_co2 else 0
                co2_max = max(self.ys_co2) if self.ys_co2 else 1000
                co2_margin = (co2_max - co2_min) * 0.1 \
                                if (co2_max - co2_min) > 0 else 100
                new_co2_max = min(3000, co2_max + co2_margin)
                self.ax_co2.set_ylim(0, new_co2_max)

                anet_min = min(self.ys_anet) if self.ys_anet else 0
                anet_max = max(self.ys_anet) if self.ys_anet else 1
                anet_margin = (anet_max - anet_min) * 0.1 \
                                if (anet_max - anet_min) > 0 else 1
                self.ax_anet.set_ylim(max(-5, anet_min - anet_margin),
                                      min(20, anet_max + anet_margin))

                self.line_co2.set_data(self.xs, self.ys_co2)
                self.line_anet.set_data(self.xs, self.ys_anet)

                lines = [self.line_anet, self.line_co2]
                labels = [line.get_label() for line in lines]
                self.ax_anet.legend(lines, labels, loc="lower right")

        return self.line_co2, self.line_anet

def calc_volume_litres(width_cm, height_cm, length_cm):
    volume_cm3 = width_cm * height_cm * length_cm
    volume_litres = volume_cm3 / 1000

    return volume_litres


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, help='Temperature in deg C',
                         default=20.)
    parser.add_argument('--no_plant_pot', action='store_true',
                        help='Turn off volume correction for plant in pot')
    parser.add_argument('--leaf_area', type=float,
                        help='Initial leaf area in cm²')
    parser.add_argument('--window_size', type=int,
                        help='Number of samples in slope estimation window',
                        default=4)
    args = parser.parse_args()

    #port = "/dev/tty.usbmodem11201" # home computer
    port = "/dev/tty.usbmodem101"   # work computer
    baud = 9600

    if args.no_plant_pot:
        lunchbox_volume = 1.0  # litres
    else:
        pot_volume = calc_volume_litres(5, 10, 5)
        lunchbox_volume = 1.0 - pot_volume  # litres


    la = args.leaf_area if args.leaf_area and args.leaf_area > 0 else 25.0
    temp = args.temp
    window_size = args.window_size

    logger = LunchboxLogger(port, baud, lunchbox_volume, temp, la, window_size,
                            measure_interval=10, timeout=1.0)
    logger.run()
