# pyqt5-gui-app/pyqt5-gui-app/README.md

# PyQt5 GUI Application

This project is a PyQt5 GUI application that demonstrates a simple home page with two buttons: RGB Feed and Thermal Feed. The application follows the Model-View-Controller (MVC) architecture, separating the concerns of data management, user interface, and application logic.

## Project Structure

```
pyqt6-gui-app
├── controllers
│   ├── __init__.py
│   ├── home_controller.py
├── models
│   ├── __init__.py
│   ├── feed_model.py
├── views
│   ├── __init__.py
│   ├── home_view.py
├── main.py
└── README.md
```

## Components

### Controllers
- **home_controller.py**: Manages the interaction between the view and the model. It handles button clicks for RGB feed and Thermal feed.

### Models
- **feed_model.py**: Contains the data and logic related to the RGB and Thermal feeds.

### Views
- **home_view.py**: Creates the GUI for the home page, including buttons for RGB feed and Thermal feed.

### Main
- **main.py**: The entry point of the application that initializes the application, creates instances of the model, view, and controller, and starts the main event loop.

## Requirements

Ensure you have PyQt5 installed in your environment to run the application. You can install it using pip:

```
pip install PyQt5
```

## Usage

To run the application, execute the `main.py` file:

```
python main.py
```

This will launch the GUI, allowing you to interact with the RGB and Thermal feed buttons.

## Additional Notes

This project serves as a foundation for further development and can be expanded with additional features and functionalities as needed.

## serial communication commands - stm board - heater controller

PS C:\Users\yyash\Documents\FRACKTAL_WORKS\VS_code\PyQt_GUI> python       
Python 3.9.13 (tags/v3.9.13:6de2ca5, May 17 2022, 16:36:42) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import serial
>>> ser = serial.Serial("COM10",115200)
>>> print(ser.write(b'$E\n')
...
... )
3
>>> ser.readline()
b'$E\n'
>>>


## SMALL PROGRAM ON PID HEATER CONTROLL FOR REFF

import time
import numpy as np
import cv2  # For processing thermal camera feed
import board
import busio
import adafruit_mlx90640  # Example thermal camera (MLX90640)
from simple_pid import PID  # Install using: pip install simple-pid

# Adjustable Parameters
TARGET_TEMPERATURE = 70.0  # Desired temperature (in °C)
PWM_MAX = 255  # Max PWM value (adjust based on hardware)
PWM_MIN = 0    # Min PWM value

# Initialize I2C for Thermal Camera (MLX90640 as an example)
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

# Initialize PID Controller
pid = PID(1.5, 0.1, 0.05, setpoint=TARGET_TEMPERATURE)
pid.output_limits = (PWM_MIN, PWM_MAX)  # Restrict heater power

# Function to get average temperature from thermal camera
def get_average_temperature():
    frame = np.zeros((24*32,))  # MLX90640 resolution (24x32)
    mlx.getFrame(frame)
    return np.mean(frame)  # Average temperature of the scene

# Function to set heater output (PWM)
def set_heater_power(power):
    power = int(power)  # Convert PID output to integer
    print(f"Heater Power: {power}")  # Replace with actual PWM control

#   Main Loop
while True:
    try:
        # Read temperature from the thermal camera
        current_temp = get_average_temperature()
        print(f"Current Temp: {current_temp:.2f} °C")

        # Compute PID output
        heater_power = pid(current_temp)

        # Apply heater control
        set_heater_power(heater_power)

        time.sleep(0.5)  # Adjust loop frequency based on system response

    except KeyboardInterrupt:
        print("Stopping heater control.")
        break
