#!/usr/bin/env python

import sys
import matplotlib
if sys.platform.startswith("win"):
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        matplotlib.use("TkAgg")  # fallback on Windows if Qt isn't available

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lunchbox_logger import LunchboxLogger
from serial_port_finder import find_usb_port
from lunchbox_logger import calc_volume_litres, calc_frustum_volume_litres

def run_plotter(temp=20.0, no_plant_pot=False, leaf_area=25.0, window_size=41,
                smoothing=True, rolling_regression=False,
                soil_resp_correction=0.0, auto_ylim=False, measure_interval=1,
                plot_duration_min=10,):

    if window_size % 2 == 0 or window_size < 5:
        raise ValueError("window_size must be an odd integer ≥ 5")

    # Setup volume and area basis
    if no_plant_pot:
        #lunchbox_volume = 1.0

        #volume_cm3 = 12 * 8 * 3 # cm
        #volume_litres = volume_cm3 / 1000
        lunchbox_volume = 0.288 # l
        area_basis = False
        la = 1.0
    else:
        pot_volume = calc_frustum_volume_litres(5.0, 3.4, 5.3)
        #lunchbox_volume = 1.0 - pot_volume # 1 l
        #volume_cm3 = 12 * 8 * 3 # cm
        #volume_litres = volume_cm3 / 1000
        lunchbox_volume = 0.288 # l
        area_basis = True
        la = leaf_area if leaf_area > 0 else 25.0

    # ls /dev/tty.*
    #port = "/dev/tty.usbmodem1101" # home computer
    #port = "/dev/tty.usbmodem1101"   # work computer
    port = find_usb_port()
    baud = 9600

    logger = LunchboxLogger(port=port, baud=baud,
                            lunchbox_volume=lunchbox_volume, temp_c=temp,
                            leaf_area_cm2=la, window_size=window_size,
                            measure_interval=measure_interval,
                            smoothing=smoothing,
                            rolling_regression=rolling_regression,
                            area_basis=area_basis,
                            soil_resp_correction=soil_resp_correction,)

    max_len = int(plot_duration_min * 60 / measure_interval)

    xs, ys_anet, ys_lower, ys_upper = [], [], [], []

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Elapsed Time (min)")
    units = "μmol m⁻² s⁻¹" if area_basis else "μmol box⁻¹ s⁻¹"
    ax.set_ylabel(f"Net assimilation rate ({units})", color="black")
    ax.set_xlim(0, plot_duration_min)
    ax.axhline(y=0.0, color="darkgrey", linestyle="--")

    line_anet, = ax.plot([], [], lw=2, color="#28b463", label="Anet")
    fill_between = None
    co2_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12,
                       verticalalignment="top", color="#8e44ad",)

    def update(frame):
        nonlocal fill_between
        data = logger.read_and_update()
        if data is None:
            return line_anet, co2_text

        elapsed_min = data["elapsed_min"]
        co2 = data["co2"]
        anet = data["anet"]
        anet_l = data["anet_lower"]
        anet_u = data["anet_upper"]

        xs.append(elapsed_min)
        ys_anet.append(anet)
        ys_lower.append(anet_l)
        ys_upper.append(anet_u)

        # Keep lists at max_len
        if len(xs) > max_len:
            xs.pop(0)
            ys_anet.pop(0)
            ys_lower.pop(0)
            ys_upper.pop(0)

        co2_text.set_text(f"CO₂ = {co2:.0f} ppm | A_net = {anet:+.2f} {units}")

        # Update x-axis limits for moving window
        ax.set_xlim(max(0, elapsed_min - plot_duration_min), elapsed_min)

        if auto_ylim:
            visible_lower = [y for x, y in zip(xs, ys_lower) \
                                if x >= max(0, elapsed_min - plot_duration_min)]
            visible_upper = [y for x, y in zip(xs, ys_upper) \
                                if x >= max(0, elapsed_min - plot_duration_min)]

            if visible_lower and visible_upper:
                anet_min = min(visible_lower)
                anet_max = max(visible_upper)
                anet_range = anet_max - anet_min

                if anet_range < 1.0:
                    mean_val = (anet_max + anet_min) / 2
                    lower = max(-10, mean_val - 1)
                    upper = mean_val + 1
                else:
                    margin = anet_range * 0.1
                    lower = max(-10, anet_min - margin)
                    upper = anet_max + margin

                ax.set_ylim(lower, upper)
            else:
                ax.set_ylim(-5, 8)

        else:
            ax.set_ylim(-5, 8)

        line_anet.set_data(xs, ys_anet)

        if fill_between:
            fill_between.remove()

        fill_between = ax.fill_between(xs, ys_lower, ys_upper, color="#0b5345",
                                       alpha=0.2, label="95% CI")

        # Update legend
        lines = [line_anet, fill_between]
        labels = [line.get_label() for line in lines if \
                        line.get_label() != "_nolegend_"]
        ax.legend(lines, labels, loc="lower right")

        return line_anet, co2_text, fill_between

    ani = animation.FuncAnimation(fig, update, interval=measure_interval * 1000,
                                  blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    logger.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, help='Temperature in deg C',
                         default=20.)
    parser.add_argument('--no_plant_pot', action='store_true',
                        help='Turn off volume correction for plant in pot')
    parser.add_argument('--leaf_area', type=float, default=25,
                        help='Initial leaf area in cm²')
    parser.add_argument('--window_size', type=int,
                        help='Number of samples in slope estimation window',
                        default=41) # must be off for smoothing filter
    parser.add_argument('--no_smoothing', action='store_true',
                        help='Turn off Savitzky-Golay and Butterworth \
                              smoothing filters')
    parser.add_argument('--rolling_regression', action='store_true',
                        help='Use rolling linear regression (statsmodels)')
    parser.add_argument('--soil_resp_correction', type=float, default=0.0,
                        help='Add soil respiration correction flux if Anet < 0')
    parser.add_argument('--auto_ylim', action='store_true',
                        help='Automatically rescale y-axis?')
    args = parser.parse_args()

    if args.window_size % 2 == 0 or args.window_size < 5:
        raise ValueError("window_size must be an odd integer ≥ 5")

    run_plotter(temp=args.temp, no_plant_pot=args.no_plant_pot,
                leaf_area=args.leaf_area, window_size=args.window_size,
                smoothing=not args.no_smoothing,
                rolling_regression=args.rolling_regression,
                soil_resp_correction=args.soil_resp_correction,
                auto_ylim=args.auto_ylim)
