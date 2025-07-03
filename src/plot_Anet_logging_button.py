import time
import qwiic_scd4x
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from scipy.stats import linregress


def main(chamber_volume, window_size, plot_window, pressure_pa,
         zero_run_duration, leaf_area_cm2_init):

    leaf_area_cm2 = [leaf_area_cm2_init]  # mutable for GUI text box updates
    print(f"Initial leaf area set to: {leaf_area_cm2[0]:.1f} cm²")

    def update_leaf_area(text):
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
            leaf_area_cm2[0] = value
            print(f"✅ Leaf area updated to {value:.1f} cm² (via GUI)")
        except ValueError:
            print("⚠️ Invalid leaf area input. Enter a positive number.")

    sensor = qwiic_scd4x.QwiicSCD4x()
    if not sensor.is_connected():
        print("Sensor not connected")
        return
    if not sensor.begin():
        print("Error initializing sensor")
        return

    # Zero-run calibration phase
    print(f"Starting zero-run calibration for {zero_run_duration} seconds.\n
            Ensure chamber is empty.")
    zero_co2_window = deque(maxlen=window_size)
    zero_time_window = deque(maxlen=window_size)
    zero_start = time.time()

    while time.time() - zero_start < zero_run_duration:
        if sensor.read_measurement():
            co2 = sensor.get_co2()
            temp = sensor.get_temperature()
            rh = sensor.get_humidity()
            co2_dry = compute_co2_dry(co2, rh, temp, pressure_pa)
            now = time.time()

            zero_co2_window.append(co2_dry)
            zero_time_window.append(now)
        time.sleep(0.5)

    if len(zero_co2_window) >= 3:
        times = np.array(zero_time_window)
        co2s = np.array(zero_co2_window)
        zero_run_slope, _, _, _, _ = linregress(times - times[0], co2s)
    else:
        zero_run_slope = 0.0

    print(f"Zero-run baseline slope: {zero_run_slope:.4f} ppm/s")

    co2_window = deque(maxlen=window_size)
    time_window = deque(maxlen=window_size)
    anet_times = deque()
    anet_values = deque()
    anet_upper = deque()
    anet_lower = deque()

    plt.ion()
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    line, = ax.plot([], [], 'g-', label="A_net")
    ci_fill = None
    ci_filled_once = False

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Net Photosynthesis (μmol m⁻² s⁻¹)")
    ax.grid(True)
    ax.legend()

    ax_text = plt.axes([0.1, 0.02, 0.3, 0.05])
    text_box = TextBox(ax_text, "Leaf Area (cm²)",
                       initial=str(leaf_area_cm2[0]))
    text_box.on_submit(update_leaf_area)

    try:
        while True:
            plt.pause(0.01)

            if sensor.read_measurement():
                co2 = sensor.get_co2()
                temp = sensor.get_temperature()
                rh = sensor.get_humidity()
                co2_dry = compute_co2_dry(co2, rh, temp, pressure_pa)
                now = time.time()

                co2_window.append(co2_dry)
                time_window.append(now)

                if len(co2_window) >= 3:
                    times = np.array(time_window)
                    co2s = np.array(co2_window)

                    res = linregress(times - times[0], co2s)
                    slope = res.slope
                    stderr = res.stderr if res.stderr is not None else 0

                    corrected_slope = slope - zero_run_slope

                    slope_upper = corrected_slope + 1.96 * stderr
                    slope_lower = corrected_slope - 1.96 * stderr

                    temp_K = temp + 273.15

                    mol_flux = ppm_to_umol_s(corrected_slope, chamber_volume,
                                             temp_K, pressure_pa)
                    mol_flux_upper = ppm_to_umol_s(slope_upper, chamber_volume,
                                                   temp_K, pressure_pa)
                    mol_flux_lower = ppm_to_umol_s(slope_lower, chamber_volume,
                                                   temp_K, pressure_pa)

                    leaf_area_m2 = leaf_area_cm2[0] / 10000.0

                    A_net = -mol_flux / leaf_area_m2
                    A_net_upper = -mol_flux_upper / leaf_area_m2
                    A_net_lower = -mol_flux_lower / leaf_area_m2

                    print(f"CO₂: {co2_dry:.1f} ppm | Temp: {temp:.1f} °C | RH: {rh:.1f}%")
                    print(f"ΔCO₂ slope: {corrected_slope:+.4f} ppm/s (± {1.96*stderr:.4f}) | A_net: {A_net:+.2f} μmol m⁻² s⁻¹")
                    print("-" * 40)

                    anet_times.append(now)
                    anet_values.append(A_net)
                    anet_upper.append(A_net_upper)
                    anet_lower.append(A_net_lower)

                    while anet_times and (now - anet_times[0]) > plot_window:
                        anet_times.popleft()
                        anet_values.popleft()
                        anet_upper.popleft()
                        anet_lower.popleft()

                    times_rel = [(t - anet_times[0]) / 60 for t in anet_times]

                    line.set_xdata(times_rel)
                    line.set_ydata(anet_values)

                    if ci_fill:
                        ci_fill.remove()

                    if not ci_filled_once:
                        ci_fill = ax.fill_between(times_rel,
                                                  list(anet_lower),
                                                  list(anet_upper),
                                                  color='seagreen', alpha=0.3,
                                                  label="95% CI")
                        ax.legend()
                        ci_filled_once = True
                    else:
                        ci_fill = ax.fill_between(times_rel,
                                                  list(anet_lower),
                                                  list(anet_upper),
                                                  color='seagreen', alpha=0.3)

                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
        plt.ioff()
        plt.show()

def ppm_to_umol_s(delta_ppm_s, volume_liters, temp_K, pressure_pa, R=8.314):
    volume_m3 = volume_liters / 1000.0
    mol_flux = (delta_ppm_s * pressure_pa * volume_m3) / (R * temp_K)
    return mol_flux

def compute_co2_dry(co2_wet_ppm, rh_percent, temp_C, pressure_pa):
    e_s = 0.611 * np.exp((17.502 * temp_C) / (temp_C + 240.97)) * 1000  # Pa
    e = rh_percent * e_s / 100.0
    return co2_wet_ppm / (1 - (e / pressure_pa))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--leaf_area', type=float,
                        help='Initial leaf area in cm^2')
    args = parser.parse_args()
    if args.leaf_area and args.leaf_area > 0:
        la_init = args.leaf_area
    else:
        la_init = 100.0

    chamber_volume = 1.2  # litres
    window_size = 6
    plot_window = 300
    pressure_pa = 101325  # Pa
    zero_run_duration = 30  # seconds

    main(chamber_volume, window_size, plot_window, pressure_pa,
         zero_run_duration, la_init)
