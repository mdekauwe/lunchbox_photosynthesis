ximport time
import threading
import qwiic_scd4x
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons
from scipy.stats import linregress


class Photosynthesis:

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
        self.zero_status_dots = 0
        # There seems to be an issue when in the light, the temp sensor
        # measures a very high value, allow the user to override this
        self.manual_temp_c = None
        self.use_dry_co2 = False
        self.dry_co2_window = np.full(window_size, np.nan)

    def run(self):

        DEG_2_K = 273.15

        thread = threading.Thread(target=self.sensor_thread, daemon=True)
        thread.start()

        try:
            while not self.stop_requested:
                plt.pause(0.05)
                with self.lock:
                    manual_temp = self.manual_temp_c
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
                        if self.use_dry_co2:
                            co2s = self.dry_co2_window
                        else:
                            co2s = co2_window
                        temps = temp_vals
                        rhs = rh_vals
                    else:
                        times = time_window[:idx]
                        if self.use_dry_co2:
                            co2s = self.dry_co2_window[:idx]
                        else:
                            co2s = co2_window[:idx]

                        temps = temp_vals[:idx]
                        rhs = rh_vals[:idx]

                    valid_mask = (~np.isnan(times)) & (~np.isnan(co2s))
                    times = times[valid_mask]
                    co2s = co2s[valid_mask]

                    if len(co2s) >= self.window_size:
                        res = linregress(times - times[0], co2s)
                        slope = res.slope
                        stderr = res.stderr or 0.0

                        corr_slope = slope - zero_slope
                        slope_upper = corr_slope + 1.96 * stderr
                        slope_lower = corr_slope - 1.96 * stderr

                        if manual_temp is not None:
                            temp_K = manual_temp + DEG_2_K
                        elif len(temps) > 0:
                            temp_K = temps[-1] + DEG_2_K
                        else:
                            temp_K = 298.15 # 25 deg

                        an_leaf = self.calc_anet(corr_slope, temp_K,
                                                 self.chamber_volume,
                                                 self.pressure_pa)
                        an_leaf_u = self.calc_anet(slope_upper, temp_K,
                                                   self.chamber_volume,
                                                   self.pressure_pa)
                        an_leaf_l = self.calc_anet(slope_lower, temp_K,
                                                   self.chamber_volume,
                                                   self.pressure_pa)

                        A_net = -an_leaf / leaf_area_m2
                        A_net_u = -an_leaf_u / leaf_area_m2
                        A_net_l = -an_leaf_l / leaf_area_m2

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
                                   (now - self.anet_times[0]) > \
                                    self.plot_window):
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
                            temp_times_rel = [(t - self.anet_times[0]) / 60 \
                                    for t in list(self.anet_times)[-min_len:]]

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

    @staticmethod
    def calc_anet(delta_ppm_s, temp_K, chamber_volume, pressure_pa):
        RGAS = 8.314
        volume_m3 = chamber_volume / 1000.0
        an_leaf = (delta_ppm_s * pressure_pa * volume_m3) / (RGAS * temp_K)

        return an_leaf # umol leaf-1 s-1

    @staticmethod
    def compute_co2_dry(co2_wet_ppm, rh_percent, temp_C, pressure_pa):
        e_s = (0.611 * np.exp((17.502 * temp_C) / (temp_C + 240.97)) * 1000)
        e = rh_percent * e_s / 100.0

        return co2_wet_ppm / (1 - (e / pressure_pa))

    @staticmethod
    def calc_vpd(temp_c, rh_percent):
        es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))  # kPa
        ea = es * (rh_percent / 100.0) # kPa

        return es - ea  # kPa

    def _setup_sensor(self):
        self.sensor = qwiic_scd4x.QwiicSCD4x()
        if not self.sensor.is_connected():
            raise RuntimeError("Sensor not connected")
        if not self.sensor.begin():
            raise RuntimeError("Error initializing sensor")

        # Disable auto self-calibration
        # The SCD40 performs automatic self-calibration by default, which
        # assumes the sensor is in ambient air for at least 1 hour per day.
        # it assumes the lowest CO2 value it sees is 400 ppm and calibrates
        # accordingly
        self.sensor.set_automatic_self_calibration_enabled(False)
        self.sensor.perform_forced_recalibration(420)

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

        self.fig.text(0.05, 0.10, "Leaf Area (cm²)", ha="left", va="bottom")
        ax_text = plt.axes([0.05, 0.05, 0.25, 0.05])
        self.text_box = TextBox(ax_text, "", initial=str(self.leaf_area_cm2[0]))
        self.text_box.on_submit(self.update_leaf_area)

        self.fig.text(0.375, 0.10, "Manual Temp (°C)", ha="left", va="bottom")
        ax_temp = plt.axes([0.375, 0.05, 0.25, 0.05])
        self.temp_box = TextBox(ax_temp, "", initial="")
        self.temp_box.on_submit(self.update_manual_temp)

        self.fig.text(0.7, 0.10, "Use Dry CO₂", ha="left", va="bottom")
        ax_check = plt.axes([0.7, 0.05, 0.25, 0.05])
        self.dry_co2_check = CheckButtons(ax_check, ['Enable'], [False])
        self.dry_co2_check.on_clicked(self.toggle_dry_co2)

    def toggle_dry_co2(self, label):
        with self.lock:
            self.use_dry_co2 = not self.use_dry_co2
        print(f"Dry CO₂ correction {'enabled' if self.use_dry_co2 else 'disabled'}")

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

    def update_manual_temp(self, text):
        try:
            if text.strip() == "":
                with self.lock:
                    self.manual_temp_c = None
                print("Using measured temp.")
            else:
                value = float(text)
                with self.lock:
                    self.manual_temp_c = value
                print(f"Manual temp set to {value:.1f} °C (override active)")
        except ValueError:
            print("Invalid input. Please enter a number or leave blank to clear.")

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
            f"  Status: Logging... ")
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
                    rh = max(0, min(100, self.sensor.get_humidity()))
                    co2_dry = self.compute_co2_dry(co2, rh, temp,
                                                   self.pressure_pa)
                    now = time.time()

                    with self.lock:
                        self.zero_data_times.append(now)
                        self.zero_data_co2.append(co2_dry)

                    elapsed = now - self.zero_data_times[0]
                    self.zero_status_dots = (self.zero_status_dots + 1) % 4
                    dots = '.' * self.zero_status_dots
                    self.status_text.set_text(f"Status: Zero run running{dots}")
                    plt.draw()

                    if elapsed >= self.zero_run_duration:
                        enough_data = False
                        with self.lock:
                            if len(self.zero_data_co2) < self.window_size:
                                print(f"Only {len(self.zero_data_co2)} zero \
                                        points. Waiting for more...")
                                self.zero_run_duration += 6
                                enough_data = False
                            else:
                                times_np = np.array(self.zero_data_times)
                                co2_np = np.array(self.zero_data_co2)
                                slope, _ = np.polyfit(times_np - times_np[0],
                                                        co2_np, 1)
                                if abs(slope) > 0.05:
                                    print(f"Warning: large zero slope = \
                                            {slope:.4f}, ignoring correction.")
                                    self.zero_slope = 0.0
                                else:
                                    print(f"Zero slope accepted = {slope:.4f}")
                                    self.zero_slope = slope

                                self.zero_data_times.clear()
                                self.zero_data_co2.clear()
                                self.zero_run_started = False
                                self.status_text.set_text("Status: Zero run \
                                                            complete")
                                plt.draw()
                                enough_data = True

                        if not enough_data:
                            time.sleep(6.0)
                            continue


            if logging:
                if self.sensor.read_measurement():
                    co2 = self.sensor.get_co2()
                    temp = self.sensor.get_temperature()
                    rh = self.sensor.get_humidity()
                    vpd = self.calc_vpd(temp, rh)
                    co2_dry = self.compute_co2_dry(co2, rh, temp,
                                                   self.pressure_pa)
                    now = time.time()

                    with self.lock:
                        idx = self.window_index
                        self.co2_window[idx] = co2          # store wet CO2
                        self.dry_co2_window[idx] = co2_dry  # store dry CO2
                        self.time_window[idx] = now
                        self.temp_values[idx] = temp
                        self.rh_values[idx] = rh

                        self.window_index = (idx + 1) % self.window_size
                        if self.window_index == 0:
                            self.window_filled = True

                        print(
                            f"CO₂: {co2_dry:.1f} | "
                            f"T: {temp:.1f} °C | RH: {rh:.1f} % | "
                            f"VPD: {vpd:.1f} kPa"
                        )
                else:
                    time.sleep(6.0)
            else:
                time.sleep(0.5)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--leaf_area', type=float,
                        help='Initial leaf area in cm²')
    args = parser.parse_args()
    la = args.leaf_area if args.leaf_area and args.leaf_area > 0 else 100.0

    # less plant size
    logger = Photosynthesis(chamber_volume=1.0, window_size=20, plot_window=300,
                            zero_run_duration=30, leaf_area_cm2_init=la)
    #logger = Photosynthesis(chamber_volume=1.2, window_size=20, plot_window=300,
    #                        zero_run_duration=30, leaf_area_cm2_init=la)
    logger.run()
