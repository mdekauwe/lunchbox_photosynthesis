#!/usr/bin/env python

"""
Forced calibration of the XENSIV PAS CO2 sensor.

Tells the sensor "the air you are in now is --ref ppm" and shifts its offset
to match. It does not need fresh air: indoors, use a value from a reference
sensor if you have one, or just pick a nominal value (e.g. 420) and treat the
readings as relative. Keep the box open, don't breathe on it, and let it sit
for a few minutes so the reading is steady.

The offset is saved in the sensor's non-volatile memory, so it survives power
cycles (use --reset to go back to factory). It only changes the absolute ppm;
A_net depends on the rate of change so is unaffected either way.
"""

import sys
import time
from serial_port_finder import find_usb_port
from xensiv_pas_co2_sensor import CO2Sensor


def show_readings(sensor, n, label):
    sensor.arm_sensor(rate_seconds=5)
    readings = [sensor.wait_for_co2(timeout_s=15) for _ in range(n)]
    print(f"{label}: {readings} ppm")
    return readings


def main(ref_ppm, save, reset):
    try:
        port = find_usb_port()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    sensor = CO2Sensor(port)
    try:
        sensor.reset_sensor()

        if reset:
            sensor.reset_forced_calibration()
            print("Saved calibration offset cleared (factory calibration).")
            show_readings(sensor, 3, "Now reading")
            return

        before = show_readings(sensor, 6, "Before calibration")
        spread = max(before[2:]) - min(before[2:])  # first readings settle
        if spread > 30:
            print(f"Warning: readings vary by {spread} ppm, the air isn't "
                  "steady so the offset will be off by roughly that much.")

        print(f"Calibrating to {ref_ppm} ppm, leave the sensor where it is "
              "(takes ~30-60 s)...")
        sensor.forced_calibration(
            ref_ppm, save=save,
            progress=lambda t: print(f"\r  {t:3.0f} s", end="", flush=True))
        print("\r  done.")

        # The new offset feeds in over a few readings, which swing wildly
        # (even below zero), so discard those before showing the result
        print("Settling...")
        sensor.arm_sensor(rate_seconds=5)
        for _ in range(5):
            sensor.wait_for_co2(timeout_s=15)
        show_readings(sensor, 4, "After calibration")
        print("Saved to sensor memory." if save else
              "Not saved: offset is lost on reset/power off.")
    finally:
        try:
            sensor.set_idle()
        except Exception:
            pass
        sensor.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Forced calibration")
    parser.add_argument("--ref", type=int, default=420,
                        help="CO2 to assign to the air the sensor is in now "
                             "(ppm, 350-1500)")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't store the offset (test only)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear the saved offset back to factory")
    args = parser.parse_args()

    main(args.ref, save=not args.no_save, reset=args.reset)
