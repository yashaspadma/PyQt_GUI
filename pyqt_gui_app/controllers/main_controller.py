from models.PrinterStatus import PrintStatus

class MainController:
    def __init__(self):
        self.heater_controller = SerialModel()
        self.print_status = PrintStatus()

    def send_command(self, command):
        self.serial_model.send_command(command)
        status = self.serial_model.get_status()
        self.print_status.update_status(status)

    def get_print_status(self):
        return self.print_status.get_status()
