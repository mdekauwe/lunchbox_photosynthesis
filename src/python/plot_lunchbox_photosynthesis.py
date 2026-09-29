#!/usr/bin/env python

import sys
import datetime
import matplotlib
if sys.platform.startswith("win"):
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        matplotlib.use("TkAgg")  # fallback on Windows if Qt isn't available

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lunchbox_logger import LunchboxLogger, air_volume_litres
from serial_port_finder import find_usb_port
from xensiv_pas_co2_sensor import MEAS_RATE_MIN_S

def run_plotter(temp=20.0, no_plant_pot=False, leaf_area=25.0, window_size=24,
                robust=True, soil_resp_correction=0.0, auto_ylim=False,
                measure_interval=MEAS_RATE_MIN_S, plot_duration_min=10,
                csv_path=None):

    # Setup volume and area basis
    lunchbox_volume = air_volume_litres(no_plant_pot)
    if no_plant_pot:
        area_basis = False
        la = 1.0
    else:
        area_basis = True
        la = leaf_area if leaf_area > 0 else 25.0

    port = find_usb_port()
    baud = 9600

    logger = LunchboxLogger(port=port, baud=baud,
                            lunchbox_volume=lunchbox_volume, temp_c=temp,
                            leaf_area_cm2=la, window_size=window_size,
                            measure_interval=measure_interval,
                            robust=robust, area_basis=area_basis,
                            soil_resp_correction=soil_resp_correction,
                            csv_path=csv_path)

    window_s = window_size * measure_interval
    print(f"Port {port} | air volume {lunchbox_volume:.3f} l | "
          f"reading every {measure_interval} s | "
          f"slope window {window_s / 60:.1f} min")

    xs_co2, ys_co2 = [], []
    xs, ys_anet, ys_lower, ys_upper = [], [], [], []

    fig, (ax_co2, ax) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 2]})
    ax_co2.set_ylabel("CO₂ (ppm)")
    line_co2, = ax_co2.plot([], [], lw=1.5, color="#8e44ad", marker=".")

    ax.set_xlabel("Elapsed Time (min)")
    units = "μmol m⁻² s⁻¹" if area_basis else "μmol box⁻¹ s⁻¹"
    ax.set_ylabel(f"Net assimilation rate ({units})", color="black")
    ax.set_xlim(0, plot_duration_min)
    ax.set_ylim(-5, 8)
    ax.axhline(y=0.0, color="darkgrey", linestyle="--")

    line_anet, = ax.plot([], [], lw=2, color="#28b463", label="Anet")
    fill_between = None
    status_text = ax_co2.text(0.01, 0.92, "Waiting for first reading...",
                              transform=ax_co2.transAxes, fontsize=12,
                              verticalalignment="top", color="#8e44ad",)

    def trim(x, *ys):
        # Drop points that have scrolled out of the plot window
        while x and x[0] < x[-1] - plot_duration_min:
            x.pop(0)
            for y in ys:
                y.pop(0)

    def update(frame):
        nonlocal fill_between
        data = logger.read_and_update()
        if data is None:
            return

        elapsed_min = data["elapsed_min"]
        co2 = data["co2"]
        xs_co2.append(elapsed_min)
        ys_co2.append(co2)
        trim(xs_co2, ys_co2)
        line_co2.set_data(xs_co2, ys_co2)
        lo, hi = min(ys_co2), max(ys_co2)
        pad = max(10, 0.1 * (hi - lo))
        ax_co2.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(max(0, elapsed_min - plot_duration_min),
                    max(plot_duration_min, elapsed_min))

        anet = data["anet"]
        if anet is None:
            status_text.set_text(f"CO₂ = {co2} ppm | filling slope window "
                                 f"{data['n']}/{window_size}")
            return

        xs.append(elapsed_min)
        ys_anet.append(anet)
        ys_lower.append(data["anet_lower"])
        ys_upper.append(data["anet_upper"])
        trim(xs, ys_anet, ys_lower, ys_upper)

        ci = (data["anet_upper"] - data["anet_lower"]) / 2
        status_text.set_text(f"CO₂ = {co2} ppm | A_net = {anet:+.2f} ± "
                             f"{ci:.2f} {units}")

        if auto_ylim:
            lower, upper = min(ys_lower), max(ys_upper)
            if upper - lower < 2.0:
                mid = (upper + lower) / 2
                lower, upper = mid - 1, mid + 1
            else:
                margin = (upper - lower) * 0.1
                lower, upper = lower - margin, upper + margin
            ax.set_ylim(max(-10, lower), upper)

        line_anet.set_data(xs, ys_anet)

        if fill_between:
            fill_between.remove()
        fill_between = ax.fill_between(xs, ys_lower, ys_upper, color="#0b5345",
                                       alpha=0.2, label="95% CI")
        ax.legend([line_anet, fill_between], ["Anet", "95% CI"],
                  loc="lower right")

    # Poll faster than the sensor rate, so readings are picked up promptly
    ani = animation.FuncAnimation(fig, update, interval=500, blit=False,
                                  cache_frame_data=False)
    plt.tight_layout()
    try:
        plt.show()
    finally:
        logger.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, help='Temperature in deg C',
                         default=20.)
    parser.add_argument('--no_plant_pot', action='store_true',
                        help='Empty box: no pot volume, flux per box')
    parser.add_argument('--leaf_area', type=float, default=25,
                        help='Leaf area in cm²')
    parser.add_argument('--interval', type=int, default=MEAS_RATE_MIN_S,
                        help='Sensor measurement interval in s (min 5)')
    parser.add_argument('--window_size', type=int, default=24,
                        help='Number of readings in the slope window '
                             '(24 × 5 s = 2 min)')
    parser.add_argument('--ols', action='store_true',
                        help='Plain least squares instead of robust fit')
    parser.add_argument('--soil_resp_correction', type=float, default=0.0,
                        help='Soil CO₂ efflux (μmol m⁻² soil s⁻¹, positive), '
                             'from calc_soil_respiration_correction.py')
    parser.add_argument('--auto_ylim', action='store_true',
                        help='Automatically rescale y-axis?')
    parser.add_argument('--save', action='store_true',
                        help='Log readings to a timestamped CSV file')
    args = parser.parse_args()

    csv_path = None
    if args.save:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"lunchbox_{stamp}.csv"
        print(f"Saving to {csv_path}")

    run_plotter(temp=args.temp, no_plant_pot=args.no_plant_pot,
                leaf_area=args.leaf_area, window_size=args.window_size,
                robust=not args.ols,
                soil_resp_correction=args.soil_resp_correction,
                auto_ylim=args.auto_ylim, measure_interval=args.interval,
                csv_path=csv_path)
