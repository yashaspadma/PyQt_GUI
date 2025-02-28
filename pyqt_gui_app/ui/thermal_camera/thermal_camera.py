import cv2

class VideoModel:
    def __init__(self):
        self.live_camera = cv2.VideoCapture(0)  # Use index or correct path for your live camera
        self.thermal_camera = cv2.VideoCapture("http://localhost:8000/video_feed")  # Use the correct URL

    def get_live_frame(self):
        ret, frame = self.live_camera.read()
        return ret, frame

    def get_thermal_frame(self):
        ret, frame = self.thermal_camera.read()
        return ret, frame

    def release_cameras(self):
        self.live_camera.release()
        self.thermal_camera.release()
        cv2.destroyAllWindows()