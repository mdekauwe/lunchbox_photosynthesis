import sys
import glob
import serial.tools.list_ports

def find_usb_port():
    if sys.platform.startswith('win'):
        # Windows: use pyserial's list_ports to find COM ports
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if 'USB' in port.description:
                return port.device
        raise RuntimeError("No USB COM port found on Windows")
    else:
        # Unix/macOS
        ports = glob.glob('/dev/tty.usbmodem*')
        if ports:
            return ports[0]
        raise RuntimeError("No /dev/tty.usbmodem* device found on Unix/macOS")
