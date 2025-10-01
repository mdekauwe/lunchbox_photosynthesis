#!/usr/bin/env python

import serial
import serial.tools.list_ports
import glob
import sys
import time
from serial_port_finder import find_usb_port

class PASCO2Calibrator:
    def __init__(self, port, baud=9600, timeout=1.5):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def send_command(self, cmd):
        self.ser.write(cmd.encode("ascii"))
        self.ser.flush()
        time.sleep(0.05)

    def read_response(self):
        resp = self.ser.readline().strip()
        return resp.decode("ascii") if resp else None

    def write_register(self, reg, data):
        cmd = f"W,{reg},{data}\n"
        self.send_command(cmd)
        resp = self.read_response()
        if resp not in ("OK", "\x06"):
            print(f"Write {reg}={data} got unexpected resp: {resp}")

    def read_register(self, reg):
        cmd = f"R,{reg}\n"
        self.send_command(cmd)
        return self.read_response()

    def calibrate_to_outside_air_ppm(self):
        print("Setting calibration reference to 400 ppm...")
        ref_val = 420
        msb = (ref_val >> 8) & 0xFF
        lsb = ref_val & 0xFF
        # 0x0D = CALIB_REF_H, 0x0E = CALIB_REF_L
        self.write_register("0D", f"{msb:02X}")
        self.write_register("0E", f"{lsb:02X}")

        print("Triggering Forced Compensation Scheme (FCS)...")
        # Read current MEAS_CFG (0x04)
        self.send_command("R,04\n")
        resp = self.read_response()
        cfg = int(resp, 16)

        # Set BOC_CFG bits [3:2] = 10b (forced compensation)
        new_cfg = (cfg & ~0b1100) | (0b10 << 2)
        self.write_register("04", f"{new_cfg:02X}")

        print("FCS started. Keep sensor in outdoor air (~420 ppm).")
        print("Waiting for 10 measurement cycles (~100 s)...")

        for i in range(10, 0, -1):
            print(f"  {i*10} s remaining...")
            time.sleep(10)

        print("FCS completed. Sensor should now be calibrated to ~420 ppm.")

        # Disable ABOC: set BOC_CFG bits [3:2] = 00b
        cfg_disabled = (cfg & ~0b1100) | (0b00 << 2)
        self.write_register("04", f"{cfg_disabled:02X}")

        print("Calibration finished and ABOC disabled. Sensor will hold this offset.")

def main():
    try:
        port = find_usb_port()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    sensor = PASCO2Calibrator(port)
    try:
        sensor.calibrate_to_outside_air_ppm()
    finally:
        sensor.close()

if __name__ == "__main__":

    main()
