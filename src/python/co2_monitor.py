#!/usr/bin/env python

import time
import qwiic_scd4x
import matplotlib.pyplot as plt
from collections import deque


def main(force_recalibrate):

    sensor = qwiic_scd4x.QwiicSCD4x()

    if not sensor.is_connected():
        print("Sensor not connected. Check wiring and power.")
        return

    if not sensor.begin():
        print("Sensor initialization failed.")
        return

    # Disable auto self-calibration
    # The SCD40 performs automatic self-calibration by default, which
    # assumes the sensor is in ambient air for at least 1 hour per day.
    # it assumes the lowest CO2 value it sees is 400 ppm and calibrates
    # accordingly...most likely, the sensor won't see "fresh air" i.e. indoors
    sensor.set_automatic_self_calibration_enabled(False)

    if force_recalibrate:
        print("\n** Waiting 3 mins for sensor to adjust in ambient air... **")
        time.sleep(180)
        print("** Performing manual calibration to 420 ppm... **")
        result = sensor.perform_forced_recalibration(420)

    print("Reading data from SCD40 sensor...\n")

    start_time = time.time()

    # Set up plot
    plt.ion()
    fig, ax = plt.subplots()
    co2_data = deque(maxlen=300)
    time_data = deque(maxlen=300)
    line, = ax.plot([], [])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("CO₂ (μmol mol⁻¹)")
    ax.grid(True)

    try:
        while True:
            if sensor.read_measurement():
                co2 = sensor.get_co2()
                temp = sensor.get_temperature()
                rh = sensor.get_humidity()

                t_now = (time.time() - start_time) / 60.0  # minutes
                co2_data.append(co2)
                time_data.append(t_now)

                # Print live
                print(
                    f"[{int(t_now):02d}:{int((t_now % 1)*60):02d}] "
                    f"CO₂: {co2:.1f} μmol mol⁻¹ | "
                    f"Temp: {temp:.1f} °C | RH: {rh:.1f} %"
                )

                # Update plot
                line.set_xdata(time_data)
                line.set_ydata(co2_data)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)

                time.sleep(2)
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nExiting sensor reader.")
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Lunchbox photosynthesis logger")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force manual recalibration to 420 ppm\
                              before starting measurements")
    args = parser.parse_args()

    main(args.recalibrate)
