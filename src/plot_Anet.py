import time
import qwiic_scd4x
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from scipy.stats import linregress


def ppm_to_umol_s(delta_ppm_s, volume_liters, temp_K, pressure_pa,
                  R=8.314):
    volume_m3 = volume_liters / 1000.0
    mol_flux = (delta_ppm_s * pressure_pa * volume_m3) / (R * temp_K)
    return mol_flux


def compute_co2_dry(co2_wet_ppm, rh_percent, temp_C, pressure_pa):
    e_s = 0.611 * np.exp((17.502 * temp_C) / (temp_C + 240.97)) * 1000
    e = rh_percent * e_s / 100.0
    return co2_wet_ppm / (1 - (e / pressure_pa))


def main(chamber_volume, window_size, plot_window, pressure_pa,
         zero_run_duration, leaf_area_cm2_init):

    sensor = qwiic_scd4x.QwiicSCD4x()
    if not sensor.is_connected():
        print("Sensor not connected")
        return
    if not sensor.begin():
        print("Error initializing sensor")
        return

    print("✅ ==Sensor ready. Use buttons to begin zero run or logging.")

    leaf_area_cm2 = [leaf_area_cm2_init]
    logging_started = [False]
    zero_run_started = [False]
    zero_slope = [0.0]
    stop_requested = [False]

    co2_window = deque(maxlen=window_size)
    time_window = deque(maxlen=window_size)
    anet_times = deque()
    anet_values = deque()
    anet_upper = deque()
    anet_lower = deque()
    zero_data_times = []
    zero_data_co2 = []

    plt.ion()
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.4)

    line, = ax.plot([], [], 'g-', label="A_net")
    ci_fill = None
    ci_filled_once = False

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Net Photosynthesis (μmol m⁻² s⁻¹)")
    ax.grid(True)
    ax.legend()

    status_text = fig.text(0.5, 0.03, "Status: Idle", ha="center")

    def update_leaf_area(text):
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
            leaf_area_cm2[0] = value
            print(f"Leaf area set to {value:.1f} cm²")
        except ValueError:
            print("Invalid input. Please enter a positive number.")

    def start_zero_run(event):
        if zero_run_started[0]:
            print("Zero run already in progress.")
            return
        if logging_started[0]:
            print("Stop logging before starting zero run.")
            return
        zero_run_started[0] = True
        zero_data_times.clear()
        zero_data_co2.clear()
        print("\nStarting zero calibration.")
        status_text.set_text("Status: Zero calibration running...")
        plt.draw()

    def start_logging(event):
        if logging_started[0]:
            print("Logging already in progress.")
            return
        if zero_run_started[0]:
            print("Wait for zero calibration to finish before logging.")
            return
        logging_started[0] = True
        print(f"\nLogging started. Leaf area = {leaf_area_cm2[0]:.1f} cm²")
        status_text.set_text(f"Status: Logging... (Leaf = {leaf_area_cm2[0]:.1f} cm²)")
        plt.draw()

    def stop_logging(event):
        print("\nStop button pressed. Exiting...")
        stop_requested[0] = True
        status_text.set_text("Status: Stopped by user")
        plt.draw()

    ax_zero = plt.axes([0.05, 0.15, 0.25, 0.075])
    zero_button = Button(ax_zero, 'Start Zero Run')
    zero_button.on_clicked(start_zero_run)

    ax_start = plt.axes([0.375, 0.15, 0.25, 0.075])
    start_button = Button(ax_start, 'Start Logging')
    start_button.on_clicked(start_logging)

    ax_stop = plt.axes([0.7, 0.15, 0.25, 0.075])
    stop_button = Button(ax_stop, 'Stop Logging')
    stop_button.on_clicked(stop_logging)

    ax_text = plt.axes([0.05, 0.03, 0.25, 0.05])
    text_box = TextBox(ax_text, "Leaf Area (cm²)",
                       initial=str(leaf_area_cm2[0]))
    text_box.on_submit(update_leaf_area)

    start_time = None

    try:
        while not stop_requested[0]:
            plt.pause(0.01)

            if zero_run_started[0]:
                if sensor.read_measurement():
                    co2 = sensor.get_co2()
                    temp = sensor.get_temperature()
                    rh = sensor.get_humidity()
                    co2_dry = compute_co2_dry(co2, rh, temp, pressure_pa)
                    now = time.time()

                    zero_data_times.append(now)
                    zero_data_co2.append(co2_dry)

                    elapsed = now - zero_data_times[0]
                    status_text.set_text(f"Zero run: {int(elapsed)} / {zero_run_duration}s")
                    plt.draw()

                    if elapsed >= zero_run_duration:
                        if len(zero_data_co2) >= 3:
                            times_np = np.array(zero_data_times)
                            co2_np = np.array(zero_data_co2)
                            slope, _ = np.polyfit(times_np - times_np[0],
                                                  co2_np, 1)
                            zero_slope[0] = slope
                            print(f"Zero run complete: {slope:.4f} ppm/s")
                        else:
                            zero_slope[0] = 0.0
                            print("Not enough zero-run data.")
                        zero_data_times.clear()
                        zero_data_co2.clear()
                        zero_run_started[0] = False
                        status_text.set_text("Status: Zero run complete")
                        plt.draw()
                time.sleep(0.5)
                continue

            if logging_started[0]:
                if start_time is None:
                    start_time = time.time()

                if sensor.read_measurement():
                    co2 = sensor.get_co2()
                    temp = sensor.get_temperature()
                    rh = sensor.get_humidity()
                    co2_dry = compute_co2_dry(co2, rh, temp, pressure_pa)
                    now = time.time()

                    co2_window.append(co2_dry)
                    time_window.append(now)

                    print(f"CO₂: {co2:.1f} wet | {co2_dry:.1f} dry | T: {temp:.1f}°C | RH: {rh:.1f}%")

                    if len(co2_window) >= 3:
                        times = np.array(time_window)
                        co2s = np.array(co2_window)
                        res = linregress(times - times[0], co2s)
                        slope = res.slope
                        stderr = res.stderr or 0.0

                        corr_slope = slope - zero_slope[0]
                        slope_upper = corr_slope + 1.96 * stderr
                        slope_lower = corr_slope - 1.96 * stderr

                        temp_K = temp + 273.15
                        leaf_area_m2 = leaf_area_cm2[0] / 10000.0

                        flux = ppm_to_umol_s(corr_slope, chamber_volume,
                                              temp_K, pressure_pa)
                        flux_u = ppm_to_umol_s(slope_upper, chamber_volume,
                                               temp_K, pressure_pa)
                        flux_l = ppm_to_umol_s(slope_lower, chamber_volume,
                                               temp_K, pressure_pa)

                        A_net = -flux / leaf_area_m2
                        A_net_u = -flux_u / leaf_area_m2
                        A_net_l = -flux_l / leaf_area_m2

                        print(f"ΔCO₂: {corr_slope:+.4f} ± {1.96*stderr:.4f} | A_net: {A_net:+.2f}")
                        print("-" * 40)

                        anet_times.append(now)
                        anet_values.append(A_net)
                        anet_upper.append(A_net_u)
                        anet_lower.append(A_net_l)

                        while anet_times and (now - anet_times[0]) > plot_window:
                            anet_times.popleft()
                            anet_values.popleft()
                            anet_upper.popleft()
                            anet_lower.popleft()

                        times_rel = [(t - anet_times[0]) / 60
                                     for t in anet_times]
                        line.set_xdata(times_rel)
                        line.set_ydata(anet_values)

                        if ci_fill:
                            ci_fill.remove()

                        ci_fill = ax.fill_between(times_rel,
                                                  list(anet_lower),
                                                  list(anet_upper),
                                                  color='seagreen', alpha=0.3)

                        if not ci_filled_once:
                            ax.legend()
                            ci_filled_once = True

                        ax.relim()
                        ax.autoscale_view()
                        ax.set_title(f"A_net (Leaf = {leaf_area_cm2[0]:.1f} cm²)")
                        plt.draw()
                else:
                    time.sleep(0.1)

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

    main(chamber_volume=1.2,
         window_size=6,
         plot_window=300,
         pressure_pa=101325,
         zero_run_duration=30,
         leaf_area_cm2_init=la_init)
