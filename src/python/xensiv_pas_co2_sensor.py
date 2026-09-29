"""
Driver for the Infineon XENSIV PAS CO2 sensor over its ASCII UART protocol.

Protocol:  write "w,<reg>,<val>\n" -> ACK (0x06) + "\n"
           read  "r,<reg>\n"       -> "<hex byte>\n"

Register map, command codes and the forced-calibration procedure follow
Infineon's reference driver (github.com/Infineon/sensor-xensiv-pasco2).

Note: all configuration registers are volatile, i.e. after a power cycle the
sensor is back in idle mode, 60 s rate, automatic baseline correction (ABOC)
on. Only the forced calibration offset survives, and only if it is saved.
"""

import serial
import time

# Registers
REG_PROD_ID = 0x00
REG_SENS_STS = 0x01
REG_MEAS_RATE_H = 0x02
REG_MEAS_RATE_L = 0x03
REG_MEAS_CFG = 0x04
REG_CO2PPM_H = 0x05
REG_CO2PPM_L = 0x06
REG_MEAS_STS = 0x07
REG_INT_CFG = 0x08
REG_PRESS_REF_H = 0x0B
REG_PRESS_REF_L = 0x0C
REG_CALIB_REF_H = 0x0D
REG_CALIB_REF_L = 0x0E
REG_SCRATCH_PAD = 0x0F
REG_SENS_RST = 0x10

# Commands written to REG_SENS_RST
CMD_SOFT_RESET = 0xA3
CMD_RESET_ABOC = 0xBC
CMD_SAVE_FCS_CALIB_OFFSET = 0xCF
CMD_RESET_FCS = 0xFC

# MEAS_CFG fields: bits [1:0] operating mode, bits [3:2] baseline offset comp.
OP_MODE_IDLE = 0b00
OP_MODE_CONTINUOUS = 0b10
BOC_DISABLE = 0b00
BOC_AUTOMATIC = 0b01
BOC_FORCED = 0b10

# SENS_STS bits
STS_SEN_RDY = 0x80
STS_ORTMP = 0x20  # temperature out of range
STS_ORVS = 0x10   # 12 V supply out of range
STS_ICCER = 0x08  # communication error (e.g. invalid value written)
STS_CLEAR_ALL = 0x07

MEAS_STS_DRDY = 0x10

ACK = 0x06
MEAS_RATE_MIN_S = 5
MEAS_RATE_MAX_S = 4095
FCS_MEAS_RATE_S = 10
SOFT_RESET_DELAY_S = 2.0
COMM_DELAY_S = 0.005


class CO2Sensor:
    def __init__(self, port, baud=9600, timeout=1.0):
        self.ser = serial.Serial(port, baudrate=baud, bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE, timeout=timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    # ---- low level -------------------------------------------------------

    def write_register(self, reg, value):
        self.ser.write(f"w,{reg:02X},{value:02X}\n".encode("ascii"))
        self.ser.flush()
        resp = self.ser.read(2)
        time.sleep(COMM_DELAY_S)
        if not resp or resp[0] != ACK:
            raise RuntimeError(f"Write 0x{reg:02X}=0x{value:02X} failed, "
                               f"response: {resp!r}")

    def read_register(self, reg):
        self.ser.write(f"r,{reg:02X}\n".encode("ascii"))
        self.ser.flush()
        resp = self.ser.read(3)
        time.sleep(COMM_DELAY_S)
        try:
            return int(resp[:2], 16)
        except ValueError:
            raise RuntimeError(f"Read 0x{reg:02X} failed, response: {resp!r}")

    def write_u16(self, reg_h, value):
        self.write_register(reg_h, (value >> 8) & 0xFF)
        self.write_register(reg_h + 1, value & 0xFF)

    def read_u16(self, reg_h):
        return (self.read_register(reg_h) << 8) | self.read_register(reg_h + 1)

    def command(self, cmd):
        self.write_register(REG_SENS_RST, cmd)

    # ---- status / configuration -----------------------------------------

    def status(self):
        """Return SENS_STS and clear its sticky error flags."""
        sts = self.read_register(REG_SENS_STS)
        if sts & (STS_ICCER | STS_ORVS | STS_ORTMP):
            self.write_register(REG_SENS_STS, STS_CLEAR_ALL)
        return sts

    def reset_sensor(self):
        """Soft reset: reloads defaults (and any saved calibration offset)."""
        self.set_idle()
        self.ser.write(f"w,{REG_SENS_RST:02X},{CMD_SOFT_RESET:02X}\n".encode())
        self.ser.flush()
        # the sensor does not reliably ACK a soft reset, so just wait it out
        time.sleep(SOFT_RESET_DELAY_S)
        self.ser.reset_input_buffer()
        sts = self.status()
        if not sts & STS_SEN_RDY:
            raise RuntimeError(f"Sensor not ready after reset (SENS_STS=0x{sts:02X})")

    def set_idle(self):
        cfg = self.read_register(REG_MEAS_CFG)
        self.write_register(REG_MEAS_CFG, cfg & ~0b11)

    def set_pressure_reference(self, pressure_pa):
        self.write_u16(REG_PRESS_REF_H, int(round(pressure_pa / 100.0)))

    def start_continuous(self, rate_seconds=MEAS_RATE_MIN_S, boc=BOC_DISABLE):
        if not (MEAS_RATE_MIN_S <= rate_seconds <= MEAS_RATE_MAX_S):
            raise ValueError(f"Rate must be {MEAS_RATE_MIN_S}-{MEAS_RATE_MAX_S} s")
        cfg = self.read_register(REG_MEAS_CFG)
        self.write_register(REG_MEAS_CFG, cfg & ~0b11)  # must be idle first
        self.write_u16(REG_MEAS_RATE_H, rate_seconds)
        cfg = (cfg & ~0b1111) | (boc << 2) | OP_MODE_CONTINUOUS
        self.write_register(REG_MEAS_CFG, cfg)

    def arm_sensor(self, rate_seconds=MEAS_RATE_MIN_S):
        # Baseline compensation off: ABOC would otherwise shift the offset
        # mid-experiment, and inside a closed box its "fresh air" assumption
        # is wrong anyway. A saved forced-calibration offset is still applied.
        self.start_continuous(rate_seconds, boc=BOC_DISABLE)

    # ---- measurements ----------------------------------------------------

    def is_data_ready(self):
        return bool(self.read_register(REG_MEAS_STS) & MEAS_STS_DRDY)

    def read_co2(self):
        """Return the latest CO2 (ppm); raises if no new value since last read."""
        if not self.is_data_ready():
            raise RuntimeError("No new CO2 data available")
        value = self.read_u16(REG_CO2PPM_H)  # reading clears DRDY
        return value - 0x10000 if value & 0x8000 else value

    def safe_read_co2(self):
        try:
            return self.read_co2()
        except Exception as e:
            print(f"CO2 read skipped: {e}")
            return None

    def wait_for_co2(self, timeout_s, poll_s=0.25):
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            if self.is_data_ready():
                return self.read_co2()
            time.sleep(poll_s)
        raise RuntimeError(f"No CO2 reading within {timeout_s} s")

    # ---- calibration -----------------------------------------------------

    def forced_calibration(self, ref_ppm, save=True, timeout_s=180,
                           progress=None):
        """
        Forced compensation (FCS): tell the sensor the air it is in right now
        is `ref_ppm`. Sensor must already be sitting in that air (e.g. outside,
        ~420 ppm) and stable. Follows Infineon's reference procedure.
        """
        if not (350 <= ref_ppm <= 1500):
            raise ValueError("Reference must be 350-1500 ppm")

        self.set_idle()
        self.write_u16(REG_MEAS_RATE_H, FCS_MEAS_RATE_S)
        self.write_u16(REG_CALIB_REF_H, ref_ppm)
        cfg = self.read_register(REG_MEAS_CFG)
        cfg = (cfg & ~0b1111) | (BOC_FORCED << 2) | OP_MODE_CONTINUOUS
        self.write_register(REG_MEAS_CFG, cfg)

        # The sensor clears BOC_CFG itself once the FCS has finished
        t0 = time.time()
        while (self.read_register(REG_MEAS_CFG) >> 2) & 0b11 == BOC_FORCED:
            if time.time() - t0 > timeout_s:
                self.set_idle()
                raise RuntimeError("Forced calibration did not finish")
            if progress:
                progress(time.time() - t0)
            time.sleep(1)

        self.set_idle()
        if save:
            self.command(CMD_SAVE_FCS_CALIB_OFFSET)

    def reset_forced_calibration(self):
        """Discard the saved forced-calibration offset (back to factory)."""
        self.set_idle()
        self.command(CMD_RESET_FCS)


def init_sensor(port):
    sensor = CO2Sensor(port)
    sensor.set_pressure_reference(101325)
    return sensor
