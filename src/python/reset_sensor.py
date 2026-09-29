#!/usr/bin/env python

from xensiv_pas_co2_sensor import CO2Sensor
from serial_port_finder import find_usb_port

port = find_usb_port()
baud=9600
timeout=1.0
sensor = CO2Sensor(port, baud, timeout)
sensor.reset_sensor()
print(f"Sensor on {port} reset, now idle.")
sensor.close()
