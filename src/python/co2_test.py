#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import argparse
import numpy as np
from xensiv_pas_co2_sensor import CO2Sensor

class CO2Logger:
    def __init__(self, port, baud, measure_interval=5, timeout=1.0,
                 plot_duration_min=10):
        self.measure_interval = measure_interval
        self.timeout = timeout
        self.plot_duration_min = plot_duration_min
        self.plot_duration_s = plot_duration_min * 60
        self.interval_ms = 1000  # update every second
        self.max_len = int(self.plot_duration_s / (self.interval_ms / 1000))

        self.sensor = CO2Sensor(port, baud, timeout)
        try:
            self.sensor.arm_sensor()
        except Exception as e:
            print(f"Failed to arm sensor: {e}")
            self.sensor.close()
            raise

        # Data buffers
        self.timestamps = deque(maxlen=self.max_len)
        self.co2_values = deque(maxlen=self.max_len)

        # Plot setup
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], lw=2, color="#5dade2", label="CO₂ (ppm)")
        self.co2_text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes,
                                     fontsize=12, verticalalignment='top',
                                     color='#8e44ad')

        self.ax.set_xlabel("Elapsed Time (min)")
        self.ax.set_ylabel("CO₂ concentration (ppm)")
        self.ax.set_ylim(300, 1000)
        self.ax.set_xlim(0, self.plot_duration_min)
        self.ax.legend(loc="upper right")
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
        elapsed_min = (current_time - self.start_time) / 60

        if (current_time - self.last_measure_time) >= self.measure_interval:
            try:
                co2 = self.sensor.read_co2()
                self.last_measure_time = current_time
                print(f"Time: {elapsed_min:.2f} min | CO₂: {co2:.1f} ppm")
                self.timestamps.append(elapsed_min)
                self.co2_values.append(co2)
                self.co2_text.set_text(f"CO₂ = {co2:.0f} ppm")
            except Exception as e:
                print(f"Read error: {e}")
                return self.line, self.co2_text

            # Update plot limits and data
            self.ax.set_xlim(
                max(0, elapsed_min - self.plot_duration_min),
                elapsed_min
            )
            ymin = min(self.co2_values) - 10
            ymax = max(self.co2_values) + 10
            self.ax.set_ylim(ymin, ymax)

            self.line.set_data(self.timestamps, self.co2_values)

        return self.line, self.co2_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default="/dev/tty.usbmodem101",
                        help='Serial port (e.g., /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=9600,
                        help='Baud rate for serial communication')
    parser.add_argument('--interval', type=int, default=5,
                        help='CO₂ measurement interval in seconds')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration of plot window in minutes')
    args = parser.parse_args()

    logger = CO2Logger(port=args.port,
                       baud=args.baud,
                       measure_interval=args.interval,
                       plot_duration_min=args.duration)
    logger.run()
