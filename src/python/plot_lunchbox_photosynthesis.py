#!/usr/bin/env python

import sys
import datetime
import matplotlib
if sys.platform.startswith("win"):
    # matplotlib.use() doesn't import the backend, so import it here to
    # find out whether any Qt binding (Qt5 or Qt6) is actually available
    try:
        import matplotlib.backends.backend_qtagg
        matplotlib.use("QtAgg")
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
                csv_path=None, low_co2=250):

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
    # A_net against CO2 for the whole run (not trimmed): as the plant draws
    # the box down this traces out its CO2 response curve
    resp_co2, resp_anet, resp_t = [], [], []

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[3, 2])
    ax_co2 = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_co2)
    ax_resp = fig.add_subplot(gs[:, 1])

    ax_co2.set_ylabel("CO₂ (ppm)")
    line_co2, = ax_co2.plot([], [], lw=1.5, color="#8e44ad", marker=".")
    ax_co2.axhline(y=low_co2, color="#c0392b", linestyle=":", lw=1)

    ax.set_xlabel("Elapsed Time (min)")
    units = "μmol m⁻² s⁻¹" if area_basis else "μmol box⁻¹ s⁻¹"
    ax.set_ylabel(f"Net assimilation rate ({units})", color="black")
    ax.set_xlim(0, plot_duration_min)
    ax.set_ylim(-5, 15)
    ax.axhline(y=0.0, color="darkgrey", linestyle="--")

    line_anet, = ax.plot([], [], lw=2, color="#28b463", label="Anet")

    ax_resp.set_xlabel("CO₂ (ppm)")
    ax_resp.set_ylabel(f"Net assimilation rate ({units})")
    ax_resp.set_title("A_net vs CO₂ (whole run)", fontsize=11)
    ax_resp.axhline(y=0.0, color="darkgrey", linestyle="--")
    ax_resp.axvspan(0, low_co2, color="#c0392b", alpha=0.08)
    ax_resp.text(low_co2, 0.98, " low CO₂ ", transform=ax_resp.get_xaxis_transform(),
                 ha="right", va="top", fontsize=9, color="#c0392b")
    resp_line, = ax_resp.plot([], [], lw=0.8, color="lightgrey", zorder=1)
    resp_pts = ax_resp.scatter([], [], c=[], cmap="viridis", s=18, zorder=2)
    cbar = fig.colorbar(resp_pts, ax=ax_resp)
    cbar.set_label("Elapsed Time (min)")
    fill_between = None
    status_text = ax_co2.set_title("Waiting for first reading...", loc="left",
                                   fontsize=12, color="#8e44ad")

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
        msg = f"CO₂ = {co2} ppm | A_net = {anet:+.2f} ± {ci:.2f} {units}"
        if co2 < low_co2:
            # The plant has drawn the box down far enough that CO2 is now
            # limiting uptake, so A_net no longer reflects the plant alone
            msg += " | CO₂ LOW: open the box to refill"
            status_text.set_color("#c0392b")
        else:
            status_text.set_color("#8e44ad")
        status_text.set_text(msg)

        resp_co2.append(data["co2_mean"])
        resp_anet.append(anet)
        resp_t.append(elapsed_min)
        resp_line.set_data(resp_co2, resp_anet)
        resp_pts.set_offsets(list(zip(resp_co2, resp_anet)))
        resp_pts.set_array(resp_t)
        resp_pts.set_clim(resp_t[0], max(resp_t[-1], resp_t[0] + 1e-3))
        x_lo, x_hi = min(resp_co2 + [low_co2]), max(resp_co2)
        ax_resp.set_xlim(x_lo - 20, x_hi + 20)
        y_lo, y_hi = min(resp_anet), max(resp_anet)
        pad = max(0.5, 0.1 * (y_hi - y_lo))
        ax_resp.set_ylim(min(y_lo, 0) - pad, max(y_hi, 0) + pad)

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
    parser.add_argument('--low_co2', type=float, default=250,
                        help='Warn when box CO₂ falls below this (ppm)')
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
                csv_path=csv_path, low_co2=args.low_co2)
