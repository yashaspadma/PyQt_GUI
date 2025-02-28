from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QInputDialog 
from PyQt5.QtCore import QSize
#from models.serial_model import SerialModel

class HomeView(QWidget):
    def __init__(self, main_window):
        super(HomeView, self).__init__()
        self.main_window = main_window

        # Load the .ui file
        self.load_ui()

    def load_ui(self):
        try:
            uic.loadUi('ControlCenter-main/src/ui/home_screen/home_screen.ui', self)
            print("UI file loaded successfully")
        except Exception as e:
            print(f"Failed to load UI file: {e}")
            raise

    def init_ui(self):
        self.setWindowTitle("SLS Camera Feed")
        self.setGeometry(100, 100, 400, 300)  # Set the window size to 400x300 (x,y width x height)
        self.setStyleSheet("background-color: red;")  # Set the background color to orange
        
        layout = QVBoxLayout()
        button_layout = QVBoxLayout()

        self.rgb_button = QPushButton("RGB Feed")
        self.thermal_button = QPushButton("Thermal Feed")
        self.serial_button = QPushButton("Serial Command")

        # Make the buttons square
        button_size = QSize(200, 200)
        self.rgb_button.setFixedSize(button_size)
        self.thermal_button.setFixedSize(button_size)
        self.serial_button.setFixedSize(button_size)

        # Change button colors
        self.rgb_button.setStyleSheet("background-color: grey; color: black;")
        self.thermal_button.setStyleSheet("background-color: grey; color: black;")
        self.serial_button.setStyleSheet("background-color: grey; color: black;")

        layout.addWidget(self.rgb_button)
        layout.addWidget(self.thermal_button)
        layout.addWidget(self.serial_button)

        self.setLayout(layout)

    def set_controller(self, controller):
        self.controller = controller
        self.rgb_button.clicked.connect(self.controller.handle_rgb_feed)
        self.thermal_button.clicked.connect(self.controller.handle_thermal_feed)
        self.serial_button.clicked.connect(self.handle_serial_command)

    def handle_serial_command(self):
        command, ok = QInputDialog.getText(self, "Serial Command", "Enter command (starting with $):")
        if ok and command:
            self.serial_model.send_command(command)