import serial
import time

class CO2Sensor:
    def __init__(self, port, baud=9600, timeout=1.5):

        self.ser = serial.Serial(port, baudrate=baud, bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE, timeout=timeout,)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def write_register(self, reg, data):
        cmd = f"W,{reg},{data}\n"
        self.ser.reset_input_buffer()
        self.ser.write(cmd.encode("ascii"))
        self.ser.flush()
        time.sleep(0.1)
        resp = self.read_response()
        if resp not in [b"OK", b"\x06"]:
            raise RuntimeError(f"Write failed, response: {resp}")

    def arm_sensor(self, rate_seconds=10):
        if not (1 <= rate_seconds <= 60):
            raise ValueError("Rate must be between 1–60 seconds")
        rate_hex = f"{rate_seconds:02X}"

        self.write_register("04", "00")  # Stop measurements
        self.write_register("02", "00")  # Disable interrupts
        self.write_register("03", rate_hex)  # Set desired measurement rate
        self.write_register("04", "02")  # Start continuous measurement
        time.sleep(rate_seconds + 2)  # wait for first fresh sample

    def send_command(self, cmd):
        self.ser.write(cmd.encode("ascii"))
        self.ser.flush()
        time.sleep(0.05)

    def read_response(self, max_retries=2, retry_delay=0.1):
        for attempt in range(max_retries + 1):
            response = self.ser.readline().strip()
            if response:
                return response
            time.sleep(retry_delay)
        raise RuntimeError("Write failed: no response after retries")

    def reset_sensor(self):
        self.write_register("07", "FF")  # Clear IRQ/status bits
        time.sleep(0.1)

    def read_co2(self, max_retries=1, retry_delay=0.2):
        if not self.is_data_ready():
            raise RuntimeError("No new CO₂ data available")

        try:
            self.send_command("R,05\n")
            msb_resp = self.read_response()
            if not msb_resp or len(msb_resp) < 2:
                raise RuntimeError(f"MSB response invalid: {msb_resp}")
            msb = int(msb_resp.decode("ascii"), 16)

            self.send_command("R,06\n")
            lsb_resp = self.read_response()
            if not lsb_resp or len(lsb_resp) < 2:
                raise RuntimeError(f"LSB response invalid: {lsb_resp}")
            lsb = int(lsb_resp.decode("ascii"), 16)

            combined = (msb << 8) | lsb
            return combined - 0x10000 if combined & 0x8000 else combined

        except Exception as e:
            if max_retries > 0 and "invalid" in str(e).lower():
                time.sleep(retry_delay)
                return self.read_co2(max_retries=max_retries-1,
                                     retry_delay=retry_delay)
            else:
                raise


    def is_data_ready(self):
        self.send_command("R,03\n")
        status_resp = self.read_response()
        if not status_resp or len(status_resp) < 2:
            return False
        status = int(status_resp.decode("ascii"), 16)
        return (status & 0x01) == 1  # Bit 0 = new data ready

    def set_pressure_reference(self, pressure_pa: int):
        pressure_hpa = int(pressure_pa / 100)
        high_byte = (pressure_hpa >> 8) & 0xFF
        low_byte = pressure_hpa & 0xFF

        # Convert register addresses to strings as expected by write_register
        self.write_register("0F", f"{high_byte:02X}")
        time.sleep(0.05)
        self.write_register("10", f"{low_byte:02X}")
        time.sleep(0.05)
