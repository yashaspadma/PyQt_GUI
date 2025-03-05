import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from controllers.controllerAPI import HomeController

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    controller = HomeController(None, window.home_page)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    #new yashas'