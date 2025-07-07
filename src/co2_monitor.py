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

    print("Reading data from SCD40 sensor...\n")

    try:
        while True:
            if sensor.read_measurement():
                co2 = sensor.get_co2()
                temp = sensor.get_temperature()
                rh = sensor.get_humidity()

                print(
                    f"CO₂: {co2:.1f} μmol mol⁻¹ | Temp: {temp:.1f} °C | "
                    f"RH: {rh:.1f} % | VPD: {vpd:.2f} kPa"
                )
                time.sleep(2)  # Adjust delay as needed
            else:
                print("Waiting for new data...")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nExiting sensor reader.")

if __name__ == "__main__":

    main()
