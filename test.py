import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# Initialize the I2C bus
i2c_bus = busio.I2C(SCL, SDA)

# Initialize the PCA9685
pca = PCA9685(i2c_bus)

# Set the PWM frequency to 60Hz (standard for general testing and LEDs)
pca.frequency = 60

print("Testing Port 8 on PCA9685... Press Ctrl+C to stop.")

try:
    while True:
        # Set port 8 to 100% duty cycle (fully ON)
        # Duty cycle uses a 16-bit value (0 to 65535 or 0xFFFF)
        pca.channels[10].duty_cycle = 0xFFFF
        print("Port 8: ON")
        time.sleep(1)

        # Set port 8 to 0% duty cycle (fully OFF)
        pca.channels[10].duty_cycle = 0
        print("Port 8: OFF")
        time.sleep(1)

except KeyboardInterrupt:
    # Safely turn off the port before exiting
    pca.channels[10].duty_cycle = 0
    print("\nTest stopped.")
