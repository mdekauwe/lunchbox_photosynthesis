#!/usr/bin/env python

"""
Force sensor calibration to a new baseline CO2 reading
"""

import argparse
import serial
import sys
import time


class ForcedCalibration:
    def __init__(self, port: str, baud: int, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout

    @staticmethod
    def _byte_to_ascii_hex(byte: int) -> str:
        """Convert a byte (0-255) to two uppercase hex ASCII chars."""
        hex_str = f"{byte:02X}"
        return hex_str

    def _write_register(self, ser, reg_addr: int, data_byte: int) -> None:
        """
        Write a single byte to a register using the ASCII UART command protocol.
        Format: w,<reg_addr_hex>,<data_hex>\n
        Expects ACK (0x06) response from sensor.
        """
        reg_ascii = self._byte_to_ascii_hex(reg_addr)
        data_ascii = self._byte_to_ascii_hex(data_byte)
        cmd = f"w,{reg_ascii},{data_ascii}\n"
        ser.write(cmd.encode("ascii"))
        resp = ser.read(1)
        if len(resp) == 0:
            raise RuntimeError("No response from sensor")
        if resp[0] != 0x06:
            raise RuntimeError(f"Sensor NAK or error response: 0x{resp[0]:02X}")

    def soft_reset(self, ser):
        self._write_register(ser, 0x10, 0xA3)
        time.sleep(0.2)

    def reset_forced_compensation(self, ser):
        self._write_register(ser, 0x10, 0xFC)
        time.sleep(0.1)

    def reset_aboc(self, ser):
        self._write_register(ser, 0x10, 0xBC)
        time.sleep(0.1)

    def run(self, baseline_ppm: int) -> None:
        """
        Perform forced calibration by writing the baseline offset ppm value
        to registers 0x0D (MSB) and 0x0E (LSB), then save with command 0xCF
        to register 0x10.
        """

        if not (350 <= baseline_ppm <= 1500):
            raise ValueError("Baseline ppm must be between 350 and 1500")

        # Convert baseline to 16-bit signed big endian
        if baseline_ppm < 0:
            baseline_ppm = (1 << 16) + baseline_ppm
        high_byte = (baseline_ppm >> 8) & 0xFF
        low_byte = baseline_ppm & 0xFF

        with serial.Serial(self.port, self.baud, timeout=self.timeout) as ser:
            time.sleep(0.2)
            self.soft_reset(ser)
            self.reset_forced_compensation(ser)
            self.reset_aboc(ser)
            self._write_register(ser, 0x0D, high_byte)
            time.sleep(0.05)
            self._write_register(ser, 0x0E, low_byte)
            time.sleep(0.05)
            self._write_register(ser, 0x10, 0xCF)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Force calib on CO2 sensor")
    parser.add_argument("baseline", type=int,
                        help="Baseline ppm value (350-1500) for forced calib")
    args = parser.parse_args()

    # ls /dev/tty.*
    #port = "/dev/tty.usbmodem1101" # home computer
    port = "/dev/tty.usbmodem1101"   # work computer
    baud = 9600
    timeout = 1.0

    try:
        C = ForcedCalibration(port, baud, timeout)
        C.run(args.baseline)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
