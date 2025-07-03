import time
import qwiic_scd4x
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

# Constants
CHAMBER_VOLUME_LITERS = 1.2
WINDOW_SIZE = 6
PLOT_WINDOW = 300  # seconds
PRESSURE = 101325  # Pa
R = 8.314  # J mol K^-1
ZERO_RUN_DURATION = 30  # seconds for zero calibration

# Control flags and state
logging_started = False
zero_run_started = False
zero_slope = 0.0
leaf_area_cm2 = [100.0]

def ppm_to_umol_s(delta_ppm_s, volume_liters, temp_K, pressure_pa):
    volume_m3 = volume_liters / 1000.0
    mol_flux = (delta_ppm_s * pressure_pa * volume_m3) / (R * temp_K)
    return mol_flux

def compute_co2_dry(co2_wet_ppm, rh_percent, temp_C, pressure_pa=PRESSURE):
    e_s = 0.611 * np.exp((17.502 * temp_C) / (temp_C + 240.97)) * 1000  # Pa
    e = rh_percent * e_s / 100.0
    return co2_wet_ppm / (1 - (e / pressure_pa))

def update_leaf_area(text):
    try:
        value = float(text)
        if value <= 0:
            raise ValueError
        leaf_area_cm2[0] = value
        print(f"✅ Leaf area set to {value:.1f} cm²")
    except ValueError:
        print("⚠️ Invalid input. Please enter a positive number for leaf area.")

def start_zero_run(event):
    global zero_run_started, logging_started
    if zero_run_started:
        print("Zero run already in progress.")
        return
    if logging_started:
        print("Stop leaf logging before starting zero run.")
        return
    zero_run_started = True
    print("\n🟡 Starting zero calibration run. Ensure chamber is empty and sealed.")
    status_text.set_text("Status: Zero run started (empty chamber)")
    plt.draw()

def start_logging(event):
    global logging_started, zero_run_started
    if logging_started:
        print("Logging already started.")
        return
    if zero_run_started:
        print("Wait for zero run to finish before starting leaf logging.")
        return
    logging_started = True
    print(f"\n🟢 Leaf logging started with leaf area = {leaf_area_cm2[0]:.1f} cm²")
    status_text.set_text(f"Status: Logging started (leaf area = {leaf_area_cm2[0]:.1f} cm²)")
    plt.draw()

def main():
    global zero_slope, zero_run_started, logging_started, status_text

    sensor = qwiic_scd4x.QwiicSCD4x()

    if not sensor.is_connected():
        print("Sensor not connected")
        return

    if not sensor.begin():
        print("Error while initializing sensor")
        return

    print("Sensor initialized. Waiting to start zero run or logging...")

    co2_window = deque(maxlen=WINDOW_SIZE)
    time_window = deque(maxlen=WINDOW_SIZE)
    anet_times = deque()
    anet_values = deque()

    plt.ion()
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    line, = ax.plot([], [], 'g-')
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Net Photosynthesis (μmol m⁻² s⁻¹)")
    ax.grid(True)

    # Status text below buttons
    status_text = fig.text(0.5, 0.02, "Status: Idle", ha="center")

    # Buttons
    ax_zero = plt.axes([0.1, 0.1, 0.3, 0.075])
    zero_button = Button(ax_zero, 'Start Zero Run')
    zero_button.on_clicked(start_zero_run)

    ax_start = plt.axes([0.55, 0.1, 0.3, 0.075])
    start_button = Button(ax_start, 'Start Logging')
    start_button.on_clicked(start_logging)

    ax_text = plt.axes([0.1, 0.02, 0.3, 0.05])
    text_box = TextBox(ax_text, "Leaf Area (cm²)", initial=str(leaf_area_cm2[0]))
    text_box.on_submit(update_leaf_area)

    start_time = None
    zero_data_times = []
    zero_data_co2 = []

    try:
        while True:
            plt.pause(0.01)

            if zero_run_started:
                if sensor.read_measurement():
                    co2 = sensor.get_co2()
                    temp = sensor.get_temperature()
                    rh = sensor.get_humidity()
                    co2_dry = compute_co2_dry(co2, rh, temp)
                    now = time.time()

                    zero_data_times.append(now)
                    zero_data_co2.append(co2_dry)

                    elapsed = now - zero_data_times[0]
                    status_text.set_text(f"Status: Zero run in progress... {int(elapsed)} / {ZERO_RUN_DURATION} s")
                    plt.draw()

                    if len(zero_data_co2) >= 3:
                        # Calculate slope over the zero run data
                        times_np = np.array(zero_data_times)
                        co2_np = np.array(zero_data_co2)
                        slope, _ = np.polyfit(times_np - times_np[0], co2_np, 1)

                        # Stop zero run after duration
                        if elapsed >= ZERO_RUN_DURATION:
                            zero_slope = slope
                            zero_run_started = False
                            status_text.set_text(f"Status: Zero run complete. Baseline slope = {zero_slope:.4f} ppm/s")
                            print(f"\n🟢 Zero calibration complete. Baseline slope: {zero_slope:.4f} ppm/s")
                            # Clear zero run data
                            zero_data_times.clear()
                            zero_data_co2.clear()
                else:
                    print(".", end="", flush=True)
                time.sleep(0.5)
                continue  # Skip rest of loop while zero run active

            if logging_started:
                if start_time is None:
                    start_time = time.time()

                if sensor.read_measurement():
                    co2 = sensor.get_co2()
                    temp = sensor.get_temperature()
                    rh = sensor.get_humidity()
                    co2_dry = compute_co2_dry(co2, rh, temp)
                    now = time.time()

                    co2_window.append(co2_dry)
                    time_window.append(now)

                    print(f"CO₂ (wet): {co2:.1f} ppm | CO₂ (dry): {co2_dry:.1f} ppm | Temp: {temp:.1f} °C | RH: {rh:.1f}%")

                    if len(co2_window) >= 3:
                        times = np.array(time_window)
                        co2s = np.array(co2_window)
                        slope, _ = np.polyfit(times - times[0], co2s, 1)

                        # Correct slope by subtracting zero baseline
                        slope_corrected = slope - zero_slope

                        temp_K = temp + 273.15
                        mol_flux = ppm_to_umol_s(slope_corrected, CHAMBER_VOLUME_LITERS, temp_K, PRESSURE)
                        leaf_area_m2 = leaf_area_cm2[0] / 10000.0
                        A_net = -mol_flux / leaf_area_m2

                        print(f" ΔCO₂ raw: {slope:+.4f} ppm/s | Corrected: {slope_corrected:+.4f} ppm/s | A_net: {A_net:+.2f} μmol m⁻² s⁻¹")
                        print("-" * 40)

                        anet_times.append(now)
                        anet_values.append(A_net)

                        while anet_times and (now - anet_times[0]) > PLOT_WINDOW:
                            anet_times.popleft()
                            anet_values.popleft()

                        times_rel = [(t - anet_times[0]) / 60 for t in anet_times]
                        line.set_xdata(times_rel)
                        line.set_ydata(anet_values)
                        ax.relim()
                        ax.autoscale_view()
                        ax.set_title(f"Net Photosynthesis (Leaf Area = {leaf_area_cm2[0]:.1f} cm²)")
                        plt.draw()
                else:
                    print(".", end="", flush=True)

            else:
                # Idle waiting state
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
