import serial
import time

class CO2Sensor:
    def __init__(self, port, baud=9600, timeout=1.0):

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
        resp = self.ser.read(2)
        if resp != b"\x06\n":
            raise RuntimeError(f"Write failed, response: {resp}")

    def arm_sensor(self):
        self.write_register("04", "00")
        self.write_register("02", "00")
        self.write_register("03", "0A")
        self.write_register("04", "02")
        time.sleep(12)

    def send_command(self, cmd):
        self.ser.write(cmd.encode("ascii"))
        self.ser.flush()
        time.sleep(0.05)

    def read_response(self, timeout=1.0):
        self.ser.timeout = timeout
        return self.ser.readline().strip()

    def read_co2(self):
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
