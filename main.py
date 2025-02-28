#there is delay when running , add logging to diagnose the delay

from PyQt5.QtWidgets import QApplication
import sys
from controllers.controllerAPI import HomeController
from models.video_model import VideoModel
from ui.main_window import HomeView

def main():
    app = QApplication(sys.argv)
    model = VideoModel()
    view = HomeView()
    controller = HomeController(model, view)

if __name__ == "__main__":
    main() 