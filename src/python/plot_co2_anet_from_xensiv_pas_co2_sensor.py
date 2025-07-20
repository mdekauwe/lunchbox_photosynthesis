#!/usr/bin/env python

"""
NB. This sensor does not record temp, so the user will need to pass this or use
the default value.
"""

import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import argparse
import numpy as np

def main(port, baud, lunchbox_volume, temp, leaf_area_cm2, window_size,
         measure_interval=10, timeout=1.0):

    DEG_2_K = 273.15
    temp_k = temp + DEG_2_K

    leaf_area_m2 = leaf_area_cm2 / 10000.0

    try:
        ser = serial.Serial(
            port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return

    try:
        arm_sensor(ser)
    except Exception as e:
        print(f"Failed to arm sensor: {e}")
        ser.close()
        return

    fig, ax_anet = plt.subplots(figsize=(10, 6))
    ax_co2 = ax_anet.twinx()

    ax_anet.set_xlabel("Elapsed Time (min)")
    ax_anet.set_ylabel("Net assimilation rate (μmol s⁻¹)", color="black")
    ax_anet.set_ylim(-1, 5)
    ax_anet.tick_params(axis="y", labelcolor="black")

    ax_co2.set_ylabel("CO₂ (ppm)", color="black")
    ax_co2.set_ylim(0, 1000)
    ax_co2.tick_params(axis="y", labelcolor="black")

    plot_duration_min = 10
    plot_duration_s = plot_duration_min * 60
    ax_anet.set_xlim(0, plot_duration_min)

    interval_ms = 60
    max_len = int(plot_duration_s / (interval_ms / 1000))

    xs = deque(maxlen=max_len)
    ys_co2 = deque(maxlen=max_len)
    ys_anet = deque(maxlen=max_len)

    co2_window = deque(maxlen=window_size)
    time_window = deque(maxlen=window_size)

    line_anet, = ax_anet.plot([], [], lw=2, color="royalblue", label="Anet")
    line_co2, = ax_co2.plot([], [], lw=2, color="seagreen", label="CO₂")

    start_time = time.time()
    last_measure_time = [0]

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        fargs=(xs, ys_co2, ys_anet, line_co2, line_anet, ax_co2, ax_anet,
               start_time, last_measure_time, ser, measure_interval,
               plot_duration_s, lunchbox_volume, temp_k, leaf_area_m2,
               plot_duration_min, co2_window, time_window),
        blit=False,
        interval=interval_ms,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()
    ser.close()

def update_plot(frame, xs, ys_co2, ys_anet, line_co2, line_anet, ax_co2,
                ax_anet, start_time, last_measure_time, ser,
                measure_interval, plot_duration_s, lunchbox_volume, temp_k,
                leaf_area_m2, plot_duration_min, co2_window, time_window):

    current_time = time.time()
    elapsed_s = current_time - start_time
    elapsed_min = elapsed_s / 60

    if (current_time - last_measure_time[0]) >= measure_interval:
        try:
            co2 = read_co2(ser)
            last_measure_time[0] = current_time
        except Exception as e:
            print(f"Read error: {e}")
            return line_co2, line_anet

        co2_window.append(co2)
        time_window.append(current_time)

        if len(co2_window) >= window_size:
            co2_array = np.array(co2_window)
            time_array = np.array(time_window)
            elapsed = time_array - time_array[0]
            slope, _ = np.polyfit(elapsed, co2_array, 1)
            anet_leaf = calc_anet(slope, lunchbox_volume, temp_k)
            anet_area = -anet_leaf / leaf_area_m2

            print(f"Time: {elapsed_min:.2f} min | "
                  f"CO₂: {co2:+.3f} ppm | "
                  f"A_net: {anet_area:+.2f} μmol m⁻² s⁻¹")

            if elapsed_s > plot_duration_s:
                shifted_xs = [x - (elapsed_min - plot_duration_min) for x in xs]
                xs.clear()
                xs.extend(shifted_xs)
                xs.append(plot_duration_min)
            else:
                xs.append(elapsed_min)

            ys_co2.append(co2)
            ys_anet.append(anet_area)

            co2_min = min(ys_co2) if ys_co2 else 0
            co2_max = max(ys_co2) if ys_co2 else 1000
            co2_margin = (co2_max - co2_min) * 0.1 if (co2_max - co2_min) > 0 else 100
            new_co2_max = min(3000, co2_max + co2_margin)
            ax_co2.set_ylim(0, new_co2_max)

            anet_min = min(ys_anet) if ys_anet else 0
            anet_max = max(ys_anet) if ys_anet else 1
            anet_margin = (anet_max - anet_min) * 0.1 if (anet_max - anet_min) > 0 else 1
            ax_anet.set_ylim(max(-5, anet_min - anet_margin), min(20, anet_max + anet_margin))

            line_co2.set_data(xs, ys_co2)
            line_anet.set_data(xs, ys_anet)

            lines = [line_anet, line_co2]
            labels = [line.get_label() for line in lines]
            ax_anet.legend(lines, labels, loc="lower right")

    return line_co2, line_anet

def write_register(ser, reg, data):
    cmd = f"W,{reg},{data}\n"
    ser.reset_input_buffer()
    ser.write(cmd.encode("ascii"))
    ser.flush()
    time.sleep(0.1)
    resp = ser.read(2)
    if resp != b"\x06\n":
        raise RuntimeError(f"Write failed, response: {resp}")

def arm_sensor(ser):
    write_register(ser, "04", "00")
    write_register(ser, "02", "00")
    write_register(ser, "03", "0A")
    write_register(ser, "04", "02")
    time.sleep(12)

def send_command(ser, cmd):
    ser.write(cmd.encode("ascii"))
    ser.flush()
    time.sleep(0.05)

def read_response(ser, timeout=1.0):
    ser.timeout = timeout
    return ser.readline().strip()

def read_co2(ser):
    send_command(ser, "R,05\n")
    msb_resp = read_response(ser)
    if not msb_resp or len(msb_resp) < 2:
        raise RuntimeError(f"MSB response invalid: {msb_resp}")
    msb = int(msb_resp.decode("ascii"), 16)

    send_command(ser, "R,06\n")
    lsb_resp = read_response(ser)
    if not lsb_resp or len(lsb_resp) < 2:
        raise RuntimeError(f"LSB response invalid: {lsb_resp}")
    lsb = int(lsb_resp.decode("ascii"), 16)

    combined = (msb << 8) | lsb
    return combined - 0x10000 if combined & 0x8000 else combined

def calc_volume_litres(width_cm, height_cm, length_cm):

    volume_cm3 = width_cm * height_cm * length_cm
    volume_litres = volume_cm3 / 1000
    return volume_litres

def calc_anet(delta_ppm_s, lunchbox_volume, temp_k):
    # Net assimilation rate (An_leaf, umol leaf-1 s-1) calculated using the
    # ideal gas law to solve for "n" amount of substance, moles of gas
    # i.e, converts ppm s-1 into umol s-1
    #
    #            delta_CO2 × p × V
    # An_leaf = -------------------
    #                  R × T
    #
    # where:
    #   delta_CO2 = rate of CO2 change (ppm s-1)
    #   p         = pressure (Pa)
    #   V         = lunchbox_volume (m3)
    #   R         = universal gas constant (J mol⁻¹ K⁻¹)
    #   T         = temperature (K)
    pressure = 101325.  # Pa
    rgas = 8.314  # J K-1 mol-1
    volume_m3 = lunchbox_volume / 1000.0  # convert litre to m3
    an_leaf = (delta_ppm_s * pressure * volume_m3) / (rgas * temp_k)

    return an_leaf  # umol leaf s-1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, help='Temperature in deg C',
                        default=18.)
    parser.add_argument('--no_plant_pot', action='store_true',
                        help='Turn off volume correction for plant in pot')
    parser.add_argument('--leaf_area', type=float,
                        help='Initial leaf area in cm²')
    parser.add_argument('--window_size', type=int,
                    help='Number of samples in slope estimation window',
                    default=10)
    args = parser.parse_args()

    port = "/dev/tty.usbmodem11201"
    baud = 9600

    if args.no_plant_pot:
        lunchbox_volume = 1.0  # litres
    else:
        pot_volume = calc_volume_litres(17.5, 12, 5)
        lunchbox_volume = 1.0 - pot_volume  # litres

    la = args.leaf_area if args.leaf_area and args.leaf_area > 0 else 25.0
    temp = args.temp
    window_size = args.window_size

    main(port, baud, lunchbox_volume, temp, la, window_size,
         measure_interval=10, timeout=1.0)
