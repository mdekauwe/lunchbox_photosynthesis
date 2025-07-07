import smbus
import time

# I2C address of the SCD40 sensor
SCD40_I2C_ADDRESS = 0x62

# I2C bus (adjust if necessary)
bus = smbus.SMBus(1)

def calculate_crc(data):
    """Calculate CRC8 checksum for the given data."""
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x31
            else:
                crc <<= 1
    return crc & 0xFF

def perform_frc(target_ppm):
    """Perform Forced Recalibration (FRC) on the SCD40 sensor."""
    # Convert target ppm to two-byte format
    target_bytes = target_ppm.to_bytes(2, 'big')
    # Calculate CRC for the target bytes
    crc = calculate_crc(target_bytes)
    # Construct the FRC command
    frc_command = [0x36, 0x2F] + list(target_bytes) + [crc]
    # Send the FRC command
    bus.write_i2c_block_data(SCD40_I2C_ADDRESS, frc_command[0], frc_command[1:])
    print(f"Sent FRC command with target {target_ppm} ppm.")

def main():
    # Initialize sensor and wait for stabilization
    print("Initializing sensor...")
    time.sleep(180)  # Wait for 3 minutes
    # Perform Forced Recalibration with a target of 400 ppm
    perform_frc(400)
    print("Forced Recalibration completed.")

if __name__ == "__main__":
    main()
