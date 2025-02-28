import cv2
from flask import Flask, Response
import threading
import numpy as np
from models.video_model import VideoModel
from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import data_to_frame, remap, cv_filter, RollingAverageFilter, connect_senxor

class HomeController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.thermal_camera = None

    def handle_rgb_feed(self):
        print("RGB feed button clicked")
        self.show_camera_feed(0)  # Assuming RGB camera is at index 0

    def handle_thermal_feed(self):
        print("Thermal feed button clicked")
        if self.thermal_camera is None:
            self.thermal_camera = ThermalCamera()
            threading.Thread(target=self.thermal_camera.start_stream, daemon=True).start()

    def show_camera_feed(self, camera_index):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            cv2.imshow(f"Camera {camera_index} Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class ThermalCamera:
    def __init__(self, roi=(0, 0, 61, 61), com_port=None):
        self.roi = roi
        self.com_port = com_port
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()

        self.temps = {"Top": 0, "Bottom": 0, "Left": 0, "Right": 0, "Center": 0}

        self.mi48, self.connected_port, _ = connect_senxor(src=self.com_port) if self.com_port else connect_senxor()

        self.mi48.set_fps(25)
        self.mi48.disable_filter(f1=True, f2=True, f3=True)
        self.mi48.set_filter_1(85)
        self.mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)
        self.mi48.set_offset_corr(0.0)
        self.mi48.set_sens_factor(100)

        self.mi48.start(stream=True, with_header=True)

        self.dminav = RollingAverageFilter(N=10)
        self.dmaxav = RollingAverageFilter(N=10)

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            self.process_frame()

    def process_frame(self):
        data, header = self.mi48.read()
        if data is None:
            return

        min_temp = self.dminav(data.min())
        max_temp = self.dmaxav(data.max())

        frame = data_to_frame(data, (80, 62), hflip=True)
        frame = np.clip(frame, min_temp, max_temp)
        frame = cv2.flip(frame, 1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        filt_frame = cv_filter(remap(frame), {'blur_ks': 3, 'd': 5, 'sigmaColor': 27, 'sigmaSpace': 27}, use_median=True, use_bilat=True, use_nlm=False)

        x1, y1, x2, y2 = self.roi
        roi_frame = filt_frame[y1:y2, x1:x2]
        roi_frame = cv2.applyColorMap(roi_frame, cv2.COLORMAP_INFERNO)
        self.draw_grid(roi_frame)
        roi_frame = cv2.resize(roi_frame, (600, 600), interpolation=cv2.INTER_LINEAR)

        temps = self.calculate_temperatures(frame, x1, y1, x2, y2)
        self.overlay_text(roi_frame, temps)

        with self.lock:
            self.latest_frame = roi_frame

    def draw_grid(self, frame):
        h, w = frame.shape[:2]
        step_w, step_h = w // 3, h // 3

        dot_length = 6
        dot_gap = 12

        for i in range(1, 3):
            x = i * step_w
            for y in range(0, h, dot_length + dot_gap):
                cv2.line(frame, (x, y), (x, min(y + dot_length, h)), (255, 255, 255), 1)

        for i in range(1, 3):
            y = i * step_h
            for x in range(0, w, dot_length + dot_gap):
                cv2.line(frame, (x, y), (min(x + dot_length, w), y), (255, 255, 255), 1)

    def calculate_temperatures(self, frame, x1, y1, x2, y2):
        w, h = x2 - x1, y2 - y1
        section_w, section_h = w // 3, h // 3

        sections = {
            "Top": frame[y1:y1+section_h, x1:x2],
            "Bottom": frame[y2-section_h:y2, x1:x2],
            "Left": frame[y1:y2, x1:x1+section_w],
            "Right": frame[y1:y2, x2-section_w:x2],
            "Center": frame[y1+section_h:y2-section_h, x1+section_w:x2-section_w]
        }

        self.temps = {name: np.mean(region) for name, region in sections.items()}
        return self.temps

    def overlay_text(self, frame, temps):
        h, w = frame.shape[:2]
        section_w, section_h = w // 3, h // 3

        positions = {
            "Top": (w // 2 - 50, section_h // 2),
            "Bottom": (w // 2 - 50, h - section_h // 2),
            "Left": (section_w // 4, h // 2),
            "Right": (w - section_w // 2 - 50, h // 2),
            "Center": (w // 2 - 50, h // 2)
        }

        for section, temp in temps.items():
            x, y = positions[section]
            cv2.putText(frame, f"{temp:.2f}C", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    def start_stream(self):
        app = Flask(__name__)

        @app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(host="0.0.0.0", port=5000, threaded=True)

    def generate_frames(self):
        while self.running:
            with self.lock:
                if self.latest_frame is None:
                    continue

                _, buffer = cv2.imencode('.jpg', self.latest_frame)
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def stop(self):
        self.running = False
        self.mi48.stop()
        cv2.destroyAllWindows()