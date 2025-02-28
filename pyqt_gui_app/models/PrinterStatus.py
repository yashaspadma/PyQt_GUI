class PrintStatus:
    def __init__(self):
        self.status = {"data": None, "feed": None}

    def update_status(self, status):
        self.status = status

    def get_status(self):
        return self.status
