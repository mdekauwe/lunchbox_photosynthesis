import time
import qwiic_scd4x

def main():

    sensor = qwiic_scd4x.QwiicSCD4x()

    if not sensor.is_connected():
        print("Sensor not connected. Check wiring and power.")
        return

    if not sensor.begin():
        print("Sensor initialization failed.")
        return

    # Enables automatic self-calibration of CO2, turn off (False) to stop
    # it doing this
    sensor.set_automatic_self_calibration_enabled(True)

    
    print("Reading data from SCD40 sensor...\n")

    start_time = time.time()

    try:
        while True:
            if sensor.read_measurement():
                co2 = sensor.get_co2()
                temp = sensor.get_temperature()
                rh = sensor.get_humidity()

                elapsed_sec = int(time.time() - start_time)
                minutes, seconds = divmod(elapsed_sec, 60)

                print(
                    f"[{minutes:02d}:{seconds:02d}] CO₂: {co2:.1f} μmol mol⁻¹ |"
                    f"Temp: {temp:.1f} °C | RH: {rh:.1f} %"
                )
                time.sleep(2)  # Adjust delay as needed
            else:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting sensor reader.")

if __name__ == "__main__":

    main()
