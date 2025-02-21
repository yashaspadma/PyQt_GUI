from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QSize

class HomeView(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SLS Camera Feed")
        self.setGeometry(100, 100, 400, 300)  # Set the window size to 400x300 (x,y width x height)

        layout = QVBoxLayout()

        self.rgb_button = QPushButton("RGB Feed")
        self.thermal_button = QPushButton("Thermal Feed")

        # Make the buttons square
        button_size = QSize(200, 200)
        self.rgb_button.setFixedSize(button_size)
        self.thermal_button.setFixedSize(button_size)

        # Change button colors
        self.rgb_button.setStyleSheet("background-color: grey; color: white;")
        self.thermal_button.setStyleSheet("background-color: grey; color: white;")


        layout.addWidget(self.rgb_button)
        layout.addWidget(self.thermal_button)

        self.setLayout(layout)

    def set_controller(self, controller):
        self.controller = controller
        self.rgb_button.clicked.connect(self.controller.handle_rgb_feed)
        self.thermal_button.clicked.connect(self.controller.handle_thermal_feed)