import time
import threading
import qwiic_scd4x
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from scipy.stats import linregress


class PhotosynthesisLogger:

    def __init__(self, chamber_volume, window_size, plot_window,
                 zero_run_duration, leaf_area_cm2_init):

        self.chamber_volume = chamber_volume
        self.window_size = window_size
        self.plot_window = plot_window
        self.pressure_pa = 101325.
        self.zero_run_duration = zero_run_duration
        self.leaf_area_cm2 = [leaf_area_cm2_init]
        self.logging_started = False
        self.zero_run_started = False
        self.stop_requested = False
        self.zero_slope = 0.0
        self.co2_window = np.full(window_size, np.nan)
        self.time_window = np.full(window_size, np.nan)
        self.temp_values = np.full(window_size, np.nan)
        self.rh_values = np.full(window_size, np.nan)
        self.window_index = 0
        self.window_filled = False
        self.anet_times = deque()
        self.anet_values = deque()
        self.anet_upper = deque()
        self.anet_lower = deque()
        self.zero_data_times = []
        self.zero_data_co2 = []
        self.start_time = None
        self.lock = threading.Lock()
        self._setup_sensor()
        self._setup_plot()

    @staticmethod
    def ppm_to_umol_s(delta_ppm_s, volume_liters, temp_K, pressure_pa, R=8.314):
        volume_m3 = volume_liters / 1000.0
        mol_flux = (delta_ppm_s * pressure_pa * volume_m3) / (R * temp_K)
        return mol_flux

    @staticmethod
    def compute_co2_dry(co2_wet_ppm, rh_percent, temp_C, pressure_pa):
        e_s = (0.611 * np.exp((17.502 * temp_C) / (temp_C + 240.97)) * 1000)
        e = rh_percent * e_s / 100.0
        return co2_wet_ppm / (1 - (e / pressure_pa))

    def _setup_sensor(self):
        self.sensor = qwiic_scd4x.QwiicSCD4x()
        if not self.sensor.is_connected():
            raise RuntimeError("Sensor not connected")
        if not self.sensor.begin():
            raise RuntimeError("Error initializing sensor")
        print("==Sensor ready. Use buttons to begin zero run or logging.")

    def _setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.4)

        self.line, = self.ax.plot([], [], 'g-', label="A_net")
        self.ci_fill = None
        self.ci_filled_once = False

        self.ax2 = self.ax.twinx()
        self.temp_line, = self.ax2.plot([], [], '-', color="#377eb8",
                                        label="Temp (°C)")
        self.rh_line, = self.ax2.plot([], [], '-', color="#ff7f00",
                                      label="RH (%)")

        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Net Photosynthesis (μmol m⁻² s⁻¹)")
        self.ax.grid(True)

        self.ax2.set_ylabel("Temp (°C) / RH (%)")
        self.ax2.set_ylim(0, 100)

        self.status_text = self.fig.text(0.5, 0.03, "Status: Idle", ha="center")

        ax_zero = plt.axes([0.05, 0.15, 0.25, 0.075])
        self.zero_button = Button(ax_zero, 'Start Zero Run')
        self.zero_button.on_clicked(self.start_zero_run)

        ax_start = plt.axes([0.375, 0.15, 0.25, 0.075])
        self.start_button = Button(ax_start, 'Start Logging')
        self.start_button.on_clicked(self.start_logging)

        ax_stop = plt.axes([0.7, 0.15, 0.25, 0.075])
        self.stop_button = Button(ax_stop, 'Stop Logging')
        self.stop_button.on_clicked(self.stop_logging)

        ax_text = plt.axes([0.05, 0.03, 0.25, 0.05])
        self.text_box = TextBox(ax_text, "Leaf Area (cm²)",
                                initial=str(self.leaf_area_cm2[0]))
        self.text_box.on_submit(self.update_leaf_area)

    def update_leaf_area(self, text):
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
            with self.lock:
                self.leaf_area_cm2[0] = value
            print(f"Leaf area set to {value:.1f} cm²")
        except ValueError:
            print("Invalid input. Please enter a positive number.")

    def start_zero_run(self, event):
        with self.lock:
            if self.zero_run_started:
                print("Zero run already in progress.")
                return
            if self.logging_started:
                print("Stop logging before starting zero run.")
                return
            self.zero_run_started = True
            self.zero_data_times.clear()
            self.zero_data_co2.clear()
        print("\nStarting zero calibration.")
        self.status_text.set_text("Status: Zero calibration running...")
        plt.draw()

    def start_logging(self, event):
        with self.lock:
            if self.logging_started:
                print("Logging already in progress.")
                return
            if self.zero_run_started:
                print("Wait for zero calibration to finish before logging.")
                return
            self.logging_started = True
        print(f"\nLogging started. Leaf area = {self.leaf_area_cm2[0]:.1f} cm²")
        self.status_text.set_text(
            f"Status: Logging... (Leaf = {self.leaf_area_cm2[0]:.1f} cm²)")
        plt.draw()

    def stop_logging(self, event):
        print("\nStop button pressed. Exiting...")
        with self.lock:
            self.stop_requested = True
        self.status_text.set_text("Status: Stopped by user")
        plt.draw()

    def sensor_thread(self):
        while True:
            with self.lock:
                if self.stop_requested:
                    break
                zero_run = self.zero_run_started
                logging = self.logging_started
            if zero_run:
                if self.sensor.read_measurement():
                    co2 = self.sensor.get_co2()
                    temp = self.sensor.get_temperature()
                    rh = self.sensor.get_humidity()
                    co2_dry = self.compute_co2_dry(co2, rh, temp,
                                                   self.pressure_pa)
                    now = time.time()

                    with self.lock:
                        self.zero_data_times.append(now)
                        self.zero_data_co2.append(co2_dry)

                    elapsed = now - self.zero_data_times[0]
                    self.status_text.set_text(
                        f"Zero run: {int(elapsed)} / {self.zero_run_duration}s"
                    )
                    plt.draw()

                    if elapsed >= self.zero_run_duration:
                        with self.lock:
                            if len(self.zero_data_co2) >= 3:
                                times_np = np.array(self.zero_data_times)
                                co2_np = np.array(self.zero_data_co2)
                                slope, _ = np.polyfit(times_np - times_np[0],
                                                     co2_np, 1)
                                self.zero_slope = slope
                                print(f"Zero run complete: {slope:.4f} ppm/s")
                            else:
                                self.zero_slope = 0.0
                                print("Not enough zero-run data.")
                            self.zero_data_times.clear()
                            self.zero_data_co2.clear()
                            self.zero_run_started = False
                            self.status_text.set_text("Status: Zero run complete")
                        plt.draw()
                time.sleep(0.5)
                continue

            if logging:
                if self.sensor.read_measurement():
                    co2 = self.sensor.get_co2()
                    temp = self.sensor.get_temperature()
                    rh = self.sensor.get_humidity()
                    co2_dry = self.compute_co2_dry(co2, rh, temp,
                                                   self.pressure_pa)
                    now = time.time()

                    with self.lock:
                        idx = self.window_index
                        self.co2_window[idx] = co2_dry
                        self.time_window[idx] = now
                        self.temp_values[idx] = temp
                        self.rh_values[idx] = rh

                        self.window_index = (idx + 1) % self.window_size
                        if self.window_index == 0:
                            self.window_filled = True

                        print(
                            f"CO₂: {co2:.1f} wet | {co2_dry:.1f} dry | "
                            f"T: {temp:.1f}°C | RH: {rh:.1f}%"
                        )
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def run(self):
        thread = threading.Thread(target=self.sensor_thread, daemon=True)
        thread.start()

        try:
            while not self.stop_requested:
                plt.pause(0.05)
                with self.lock:
                    logging = self.logging_started
                    zero_run = self.zero_run_started
                    co2_data = (self.co2_window.copy(),
                                self.time_window.copy(),
                                self.temp_values.copy(),
                                self.rh_values.copy(),
                                self.window_filled,
                                self.window_index)
                    anet_times = list(self.anet_times)
                    anet_values = list(self.anet_values)
                    anet_upper = list(self.anet_upper)
                    anet_lower = list(self.anet_lower)
                    zero_slope = self.zero_slope
                    leaf_area_m2 = self.leaf_area_cm2[0] / 10000.0

                if zero_run:
                    continue

                if logging:
                    (co2_window, time_window, temp_vals,
                     rh_vals, filled, idx) = co2_data

                    if filled:
                        times = time_window
                        co2s = co2_window
                        temps = temp_vals
                        rhs = rh_vals
                    else:
                        times = time_window[:idx]
                        co2s = co2_window[:idx]
                        temps = temp_vals[:idx]
                        rhs = rh_vals[:idx]

                    valid_mask = (~np.isnan(times)) & (~np.isnan(co2s))
                    times = times[valid_mask]
                    co2s = co2s[valid_mask]

                    if len(co2s) >= 3:
                        res = linregress(times - times[0], co2s)
                        slope = res.slope
                        stderr = res.stderr or 0.0

                        corr_slope = slope - zero_slope
                        slope_upper = corr_slope + 1.96 * stderr
                        slope_lower = corr_slope - 1.96 * stderr

                        if len(temps) > 0:
                            temp_K = temps[-1] + 273.15
                        else:
                            temp_K = 298.15

                        flux = self.ppm_to_umol_s(corr_slope,
                                                  self.chamber_volume,
                                                  temp_K, self.pressure_pa)
                        flux_u = self.ppm_to_umol_s(slope_upper,
                                                    self.chamber_volume,
                                                    temp_K, self.pressure_pa)
                        flux_l = self.ppm_to_umol_s(slope_lower,
                                                    self.chamber_volume,
                                                    temp_K, self.pressure_pa)

                        A_net = -flux / leaf_area_m2
                        A_net_u = -flux_u / leaf_area_m2
                        A_net_l = -flux_l / leaf_area_m2

                        print(
                            f"ΔCO₂: {corr_slope:+.4f} ± {1.96*stderr:.4f} | "
                            f"A_net: {A_net:+.2f}"
                        )
                        print("-" * 40)

                        now = time.time()
                        with self.lock:
                            self.anet_times.append(now)
                            self.anet_values.append(A_net)
                            self.anet_upper.append(A_net_u)
                            self.anet_lower.append(A_net_l)

                            while (self.anet_times and
                                   (now - self.anet_times[0]) > self.plot_window):
                                self.anet_times.popleft()
                                self.anet_values.popleft()
                                self.anet_upper.popleft()
                                self.anet_lower.popleft()

                        times_rel = [(t - self.anet_times[0]) / 60
                                     for t in self.anet_times]
                        self.line.set_xdata(times_rel)
                        self.line.set_ydata(self.anet_values)

                        if self.ci_fill:
                            self.ci_fill.remove()

                        self.ci_fill = self.ax.fill_between(
                            times_rel,
                            list(self.anet_lower),
                            list(self.anet_upper),
                            color='seagreen', alpha=0.3)

                        min_len = min(len(self.anet_times),
                                      len(temps), len(rhs))
                        if min_len > 0:
                            temp_times_rel = [
                                (t - self.anet_times[0]) / 60
                                for t in list(self.anet_times)[-min_len:]
                            ]
                            self.temp_line.set_xdata(temp_times_rel)
                            self.temp_line.set_ydata(list(temps)[-min_len:])

                            self.rh_line.set_xdata(temp_times_rel)
                            self.rh_line.set_ydata(list(rhs)[-min_len:])
                        else:
                            self.temp_line.set_xdata([])
                            self.temp_line.set_ydata([])
                            self.rh_line.set_xdata([])
                            self.rh_line.set_ydata([])

                        self.ax.relim()
                        self.ax.autoscale_view()
                        self.ax2.relim()
                        self.ax2.autoscale_view()

                        if not self.ci_filled_once:
                            # Combine all lines in one legend at upper left
                            lines = [self.line, self.temp_line, self.rh_line]
                            labels = [line.get_label() for line in lines]
                            self.ax.legend(lines, labels, loc='upper left')
                            self.ci_filled_once = True

                        plt.draw()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            plt.ioff()
            plt.show()
            print("Exited cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaf_area', type=float,
                        help='Initial leaf area in cm²')
    args = parser.parse_args()
    la_init = args.leaf_area if args.leaf_area and args.leaf_area > 0 else 100.0

    logger = PhotosynthesisLogger(
        chamber_volume=1.2,
        window_size=6,
        plot_window=300,
        zero_run_duration=30,
        leaf_area_cm2_init=la_init
    )
    logger.run()
