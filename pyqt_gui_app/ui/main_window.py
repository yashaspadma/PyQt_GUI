from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStackedWidget
from ui.home_page.home_page import HomeView
from ui.heater_controller.heater_controller import SerialModel
from ui.thermal_camera.thermal_camera import VideoModel
import traceback

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)

        # Load sub UIs based on configuration
        self.load_home_page()
        #self.load_loading_screen()
        self.load_heater_controller()
        self.load_thermal_camera()
        #self.switch_screen(self.loading_screen)

        # Adjust the size of the main window to fit its contents
        self.adjustSize()

    def load_home_page(self):
        self.home_page = HomeView(self)
        self.stacked_widget.addWidget(self.home_page)
        self.switch_to_home_()

    #def load_loading_screen(self):
    #    self.loading_screen = (self)
    #    self.stacked_widget.addWidget(self.loading_screen)
    
    def load_heater_controller(self):
        self.heater_controller = SerialModel(self)
        self.stacked_widget.addWidget(self.heater_controller)

    def load_thermal_camera(self):
        self.thermal_camera = VideoModel(self)
        self.stacked_widget.addWidget(self.thermal_camera)

    def switch_screen(self, widget):
        print(f"Switching to screen: {widget}")
        traceback.print_stack()  # Print the call stack
        self.stacked_widget.setCurrentWidget(widget)
#        self.adjustSize()  # Adjust size after switching screens

    def switch_to_home_(self):
        self.switch_screen(self.home_page)

    #def switch_to_network_settings(self):
    #    self.switch_screen(self.network_settings)

