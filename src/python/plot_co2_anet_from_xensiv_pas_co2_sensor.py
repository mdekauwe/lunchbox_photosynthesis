#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import argparse
import numpy as np
import serial
from scipy.signal import savgol_filter, butter, filtfilt
import statsmodels.api as sm
import pandas as pd

from xensiv_pas_co2_sensor import CO2Sensor

class LunchboxLogger:
    def __init__(self, port, baud, lunchbox_volume, temp_c, leaf_area_cm2,
                 window_size, measure_interval=10, timeout=1.0,
                 plot_duration_min=10, smoothing=True,
                 rolling_regression=False):

        self.temp_k = temp_c + 273.15
        self.pressure = 101325.  # Pa
        self.leaf_area_m2 = leaf_area_cm2 / 10000.0
        self.lunchbox_volume = lunchbox_volume
        self.window_size = window_size
        self.measure_interval = measure_interval
        self.plot_duration_min = plot_duration_min
        self.plot_duration_s = plot_duration_min * 60
        self.interval_ms = 1000#200#60
        self.max_len = int(self.plot_duration_s / (self.interval_ms / 1000))

        # Data buffers
        self.xs = deque(maxlen=self.max_len)
        self.ys_anet = deque(maxlen=self.max_len)
        self.ys_anet_lower = deque(maxlen=self.max_len)
        self.ys_anet_upper = deque(maxlen=self.max_len)
        self.co2_window = deque(maxlen=window_size)
        self.time_window = deque(maxlen=window_size)
        self.anet_fill = None

        # Setup sensor
        self.sensor = CO2Sensor(port, baud, timeout)
        try:
            self.sensor.reset_sensor()
            self.sensor.set_pressure_reference(101325)  # default was 900 hpa
            self.sensor.arm_sensor(rate_seconds=self.measure_interval)
        except Exception as e:
            print(f"Failed to arm sensor: {e}")
            self.sensor.close()
            raise
        self.smoothing = smoothing
        self.rolling_regression = rolling_regression

        # Plot setup
        self.fig, self.ax_anet = plt.subplots(figsize=(12, 6))
        self._setup_axes()

        self.line_anet, = self.ax_anet.plot([], [], lw=2, color="#28b463", \
                            label="Anet")
        self.co2_text = self.ax_anet.text(
                            0.02, 0.95, "", transform=self.ax_anet.transAxes,
                            fontsize=12, verticalalignment='top',
                            color='#8e44ad')
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
        self.ax_anet.set_ylim(-5, 20)
        self.ax_anet.tick_params(axis="y", labelcolor="black")
        self.ax_anet.set_xlim(0, self.plot_duration_min)
        self.ax_anet.axhline(y=0.0, color='darkgrey', linestyle='--')

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
                self.co2_text.set_text(f"CO₂ = {co2:.0f} ppm")
            except Exception as e:
                print(f"Read error: {e}")
                return self.line_anet, self.co2_text

            self.co2_window.append(co2)
            self.time_window.append(current_time)

            if len(self.co2_window) >= self.window_size:
                co2_array = np.array(self.co2_window)
                time_array = np.array(self.time_window)
                elapsed = time_array - time_array[0]
                elapsed = np.round(elapsed, 2)
                elapsed -= elapsed.mean()


                if self.smoothing and len(co2_array) >= self.window_size:
                    # Apply Savitzky-Golay filter to smooth CO2 values, but
                    # preserve peaks and slopes, i.e., stabilize deltaCO2 vs
                    # time slope. Using polyorder=2 vs 3 seems to have
                    # smoothed out the high-frequency noise more gently
                    co2_array_smooth = savgol_filter(co2_array,
                                                     window_length=15,
                                                     polyorder=2)

                    # Use a Butterworth low-pass to remove the remaining
                    # cyclical noise that seems to be associated with the
                    # co2 sensor, i.e., when we don't have a plant in the box

                    # sampling frequency in Hz (e.g., 0.5 Hz if
                    # measure_interval=2)
                    fs = 1 / self.measure_interval

                    # Hz — adjust based on oscillation frequency you want to
                    # filter out
                    #cutoff = 0.01 # 1 cycle every 100 seconds

                    #cutoff = 0.025
                    cutoff = 0.05 # 1 cycle every 20 second

                    # could drop the order=4
                    # order = 6 needs a window size of 25
                    # order = 4 works fine with 21
                    co2_array_filter = butter_lowpass_filter(co2_array_smooth,
                                                             cutoff, fs,
                                                             order=4)

                    #co2_array_filter = butter_bandpass_filter(co2_array_smooth,
                    #                      lowcut=0.003,  # ~5.5 min
                    #                      highcut=0.05,  # ~20 sec
                    #                      fs=1.0,        # 1 Hz sampling
                    #                      order=4)
                else:
                    co2_array_filter = co2_array

                if self.rolling_regression:
                    # Rolling linear regression
                    X = sm.add_constant(elapsed)
                    #model = sm.OLS(co2_array_filter, X)
                    # less sensitive to outliers

                    model = sm.RLM(co2_array_filter, X,
                                   M=sm.robust.norms.HuberT())
                    results = model.fit()

                    slope = results.params[1]
                    intercept = results.params[0]
                    print(f"slope: {slope:.5f} ppm/s")
                    # Calculate standard error of slope for 95% CI
                    stderr = results.bse[1] if results.bse.size > 1 else 0
                else:

                    (p, residuals, rank,
                     singular_values,
                     rcond) = np.polyfit(elapsed, co2_array_filter, 1,
                                         full=True)

                    slope = p[0]
                    intercept = p[1]

                    n = len(elapsed)
                    if n > 2 and residuals.size > 0:
                        residual_var = residuals[0] / (n - 2)
                        x_var = np.var(elapsed, ddof=1)
                        stderr = np.sqrt(residual_var / (n * x_var))
                    else:
                        stderr = 0

                slope_upper = slope + 1.96 * stderr
                slope_lower = slope - 1.96 * stderr

                anet_leaf = self.calc_anet(slope)
                anet_leaf_u = self.calc_anet(slope_upper)
                anet_leaf_l = self.calc_anet(slope_lower)

                anet_area = -anet_leaf / self.leaf_area_m2
                anet_area_u = -anet_leaf_u / self.leaf_area_m2
                anet_area_l = -anet_leaf_l / self.leaf_area_m2

                print(f"Time: {elapsed_min:.2f} min | "
                      f"CO₂: {co2:.3f} ppm | "
                      f"A_net: {anet_area:+.2f} μmol m⁻² s⁻¹")

                self.xs.append(elapsed_min)
                self.ys_anet.append(anet_area)
                self.ys_anet_lower.append(anet_area_l)
                self.ys_anet_upper.append(anet_area_u)

                # Smooth Anet data for plotting
                #anet_series = pd.Series(self.ys_anet)
                #anet_smooth = anet_series.rolling(window=5, center=True).mean()

                # Set x-axis limits for moving window
                self.ax_anet.set_xlim(
                    max(0, elapsed_min - self.plot_duration_min),
                    elapsed_min)

                if self.ys_anet_lower and self.ys_anet_upper:
                    # Filter values based on visible x range
                    x_min = max(0, elapsed_min - self.plot_duration_min)

                    visible_vals_l = [y for x, y in zip(self.xs, \
                                            self.ys_anet_lower) if x >= x_min]
                    visible_vals_u = [y for x, y in zip(self.xs, \
                                            self.ys_anet_upper) if x >= x_min]

                    if visible_vals_l and visible_vals_u:
                        anet_min = min(visible_vals_l)
                        anet_max = max(visible_vals_u)
                        anet_range = anet_max - anet_min

                        if anet_range < 1.0:
                            mean_val = (anet_max + anet_min) / 2
                            self.ax_anet.set_ylim(mean_val - 1, mean_val + 1)
                        else:
                            margin = anet_range * 0.1
                            self.ax_anet.set_ylim(anet_min - margin,
                                                  anet_max + margin)
                    else:
                        self.ax_anet.set_ylim(-5, 8)

                # Update plot with raw data
                self.line_anet.set_data(self.xs, self.ys_anet)

                # Update plot with smoothed Anet
                #self.line_anet.set_data(self.xs, anet_smooth)

                # Remove previous envelope if it exists
                if self.anet_fill is not None:
                    self.anet_fill.remove()

                # Fill between the raw upper and lower bounds (no smoothing)
                self.anet_fill = self.ax_anet.fill_between(
                    list(self.xs),
                    list(self.ys_anet_lower),
                    list(self.ys_anet_upper),
                    color='#0b5345',
                    alpha=0.2,
                    label='95% CI'
                )

                # Update legend
                lines = [self.line_anet, self.anet_fill]
                labels = [line.get_label() for line in lines\
                                if line.get_label() != '_nolegend_']
                self.ax_anet.legend(lines, labels, loc="lower right")

        return self.line_anet, self.co2_text, self.anet_fill

def calc_volume_litres(width_cm, height_cm, length_cm):
    volume_cm3 = width_cm * height_cm * length_cm
    volume_litres = volume_cm3 / 1000

    return volume_litres

def butter_lowpass_filter(data, cutoff, fs, order=4):
    # suppress high-frequency sensor noise but keep long-period oscillations
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)

    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    # Band-pass filter to exclude both high- and low-frequency components, with
    # the aim of keeping only the plant signal changes in co2
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, data)

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
                        default=31) # must be off for smoothing filter
    parser.add_argument('--no_smoothing', action='store_true',
                        help='Turn off Savitzky-Golay and Butterworth \
                              smoothing filters')
    parser.add_argument('--rolling_regression', action='store_true',
                        help='Use rolling linear regression (statsmodels)')
    args = parser.parse_args()

    if args.window_size % 2 == 0 or args.window_size < 5:
        raise ValueError("window_size must be an odd integer ≥ 5")

    # ls /dev/tty.*
    #port = "/dev/tty.usbmodem1101" # home computer
    port = "/dev/tty.usbmodem1101"   # work computer
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
                            measure_interval=1, timeout=1.0,
                            smoothing=not args.no_smoothing,
                            rolling_regression=args.rolling_regression)
    logger.run()
