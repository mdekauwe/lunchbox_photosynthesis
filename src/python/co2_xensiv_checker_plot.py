#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import argparse
import numpy as np
from scipy.signal import savgol_filter

from xensiv_pas_co2_sensor import CO2Sensor

class CO2SmoothingViewer:
    def __init__(self, port, baud, window_size, measure_interval=10,
                 timeout=1.0, plot_duration_min=10):

        self.window_size = window_size
        self.measure_interval = measure_interval
        self.plot_duration_min = plot_duration_min
        self.plot_duration_s = plot_duration_min * 60
        self.interval_ms = 60
        self.max_len = int(self.plot_duration_s / (self.interval_ms / 1000))

        self.co2_raw = deque(maxlen=self.max_len)
        self.co2_smooth = deque(maxlen=self.max_len)
        self.time_points = deque(maxlen=self.max_len)
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
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line_raw, = self.ax.plot([], [], lw=2, color="red",
                                      label="Raw CO₂")
        self.line_smooth, = self.ax.plot([], [], lw=2, color="#1f77b4",
                                         label="Smoothed CO₂")
        self.ax.set_xlabel("Elapsed Time (min)")
        self.ax.set_ylabel("CO₂ concentration (ppm)")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(300, 600)
        self.ax.legend()
        self.co2_text = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')

        self.start_time = time.time()
        self.last_measure_time = 0

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, blit=False,
                                      interval=self.interval_ms,
                                      cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        self.sensor.close()

    def update(self, frame):
        current_time = time.time()
        elapsed_s = current_time - self.start_time
        elapsed_min = elapsed_s / 60

        # Read new data every measure_interval seconds
        if (current_time - self.last_measure_time) >= self.measure_interval:
            try:
                co2 = self.sensor.read_co2()

                # DEBUG: Simulate CO2 values
                # co2 = 400 + 30 * np.sin(current_time / 30.0)

                self.last_measure_time = current_time
                self.co2_text.set_text(f"CO₂ = {co2:.0f} ppm")

                # Append new data
                self.time_points.append(elapsed_min)
                self.co2_raw.append(co2)

                self.co2_window.append(co2)
                self.time_window.append(current_time)

                if len(self.co2_window) >= self.window_size:
                    co2_array = np.array(self.co2_window)
                    co2_smooth = savgol_filter(co2_array,
                                               window_length=self.window_size,
                                               polyorder=2)
                    self.co2_smooth.append(co2_smooth[-1])
                else:
                    self.co2_smooth.append(co2)

            except Exception as e:
                print(f"Read error: {e}")
                return self.line_raw, self.line_smooth, self.co2_text

        # Always update plot using current buffer contents
        if len(self.time_points) > 1:
            #print(f"Raw buffer length: {len(self.co2_raw)}")

            self.line_raw.set_data(list(self.time_points), list(self.co2_raw))
            self.line_smooth.set_data(list(self.time_points), list(self.co2_smooth))

            co2_min = min(self.co2_raw)
            co2_max = max(self.co2_raw)
            margin = (co2_max - co2_min) * 0.1 if co2_max > co2_min else 50

            self.ax.set_ylim(co2_min - margin, co2_max + margin)
            self.ax.set_xlim(max(0, elapsed_min - self.plot_duration_min),
                             elapsed_min)

        return self.line_raw, self.line_smooth, self.co2_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int,
                        help='Number of samples in smoothing window (odd)',
                        default=21) # must be off for smoothing filter
    args = parser.parse_args()

    if args.window_size % 2 == 0 or args.window_size < 5:
        raise ValueError("window_size must be an odd integer ≥ 5")

    port = "/dev/tty.usbmodem1101"
    baud = 9600

    viewer = CO2SmoothingViewer(port, baud,
                                window_size=args.window_size,
                                measure_interval=2,
                                timeout=1.0)
    viewer.run()
