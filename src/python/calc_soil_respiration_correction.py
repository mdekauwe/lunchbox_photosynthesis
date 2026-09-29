#!/usr/bin/env python

"""
Estimate soil respiration: put the pot with soil only (no plant, or with the
plant covered/in the dark) in the closed box. CO2 then only rises, and the
rate gives the soil efflux per m² of soil surface. Pass the printed value to
plot_lunchbox_photosynthesis.py --soil_resp_correction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lunchbox_logger import (LunchboxLogger, air_volume_litres, POT_TOP_CM)
from serial_port_finder import find_usb_port
from xensiv_pas_co2_sensor import MEAS_RATE_MIN_S


def print_final_stats(efflux_values, ignore_initial_min):
    print("\nStopping measurements.")
    if not efflux_values:
        print(f"No estimates after the first {ignore_initial_min} min, "
              "run for longer.")
        return
    vals = np.array(efflux_values)
    print(f"Soil respiration (n={len(vals)} windows): "
          f"{np.mean(vals):.3f} ± {np.std(vals):.3f} μmol m⁻² s⁻¹")
    print(f"Use: --soil_resp_correction {max(np.median(vals), 0):.3f}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, default=20.,
                        help='Temperature in deg C')
    parser.add_argument('--window_size', type=int, default=24,
                        help='Number of readings in the slope window')
    parser.add_argument('--ignore_initial_min', type=float, default=2.0,
                        help='Ignore estimates while the box settles')
    args = parser.parse_args()

    soil_area_m2 = (POT_TOP_CM / 100.0) ** 2

    # Flux per box (area_basis=False) and no correction; uptake is positive,
    # so soil efflux = -flux
    logger = LunchboxLogger(port=find_usb_port(), baud=9600,
                            lunchbox_volume=air_volume_litres(),
                            temp_c=args.temp, leaf_area_cm2=1.0,
                            window_size=args.window_size,
                            measure_interval=MEAS_RATE_MIN_S, area_basis=False)

    efflux_values = []
    xs, ys = [], []

    fig, ax = plt.subplots()
    line, = ax.plot([], [], ".", color="grey", label="per window")
    line_mean, = ax.plot([], [], lw=2, color="#a04000", label="running mean")
    ax.set_xlabel("Elapsed Time (min)")
    ax.set_ylabel("Soil respiration (μmol m⁻² s⁻¹)")
    ax.axvline(args.ignore_initial_min, color="darkgrey", linestyle="--")
    ax.legend(loc="upper right")

    def update(frame):
        data = logger.read_and_update()
        if data is None or data["anet"] is None:
            return

        efflux = -data["anet"] / soil_area_m2
        elapsed = data["elapsed_min"]
        if elapsed > args.ignore_initial_min:
            efflux_values.append(efflux)
        xs.append(elapsed)
        ys.append(efflux)

        mean = np.mean(efflux_values) if efflux_values else float("nan")
        print(f"Elapsed {elapsed:.2f} min | CO₂ {data['co2']} ppm | "
              f"Soil resp {efflux:.3f} | Mean {mean:.3f} μmol m⁻² s⁻¹")

        line.set_data(xs, ys)
        n_ignored = len(xs) - len(efflux_values)
        running = np.cumsum(efflux_values) / np.arange(1, len(efflux_values) + 1)
        line_mean.set_data(xs[n_ignored:], running)
        ax.relim()
        ax.autoscale_view()

    ani = animation.FuncAnimation(fig, update, interval=500,
                                  cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print_final_stats(efflux_values, args.ignore_initial_min)
        logger.close()
