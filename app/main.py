import cv2
from datetime import datetime
from app.ml.yolo_v3 import YoloV3

import kivy

kivy.require('1.0.6')
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.clock import Clock
from kivymd.uix.screen import MDScreen
from kivymd.uix.button.button import MDIconButton
from kivymd.uix.label.label import MDLabel
from kivy.uix.image import Image
from kivy.uix.anchorlayout import AnchorLayout
from kivy.graphics.texture import Texture
from constants import Constants


class WildlifeCameraTrapApp(MDApp):
    boxes = []
    class_names = []
    confidences = []
    should_detect = False
    recording = False
    original_recorder = None
    marked_recorder = None
    last_object_detected_time = None
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    def build(self):
        self._load_model()
        return self._create_layout()

    # Run continuously to get webcam feed
    def update(self, *args):
        _, frame = self.capture.read()
        elapsed_time = (datetime.now() - self.starting_time).total_seconds()
        current_fps = self.frame_id / elapsed_time
        original_frame = frame.copy()
        self.frame_id += 1
        if self.should_detect:
            boxes, confidences, class_names = self.yolo_v3.detect_objects(frame)
            # when objects are detected
            if len(boxes):
                if not self.recording:
                    self.recording = True
                    self.original_recorder = cv2.VideoWriter(
                        f'./app/data/{self.last_object_detected_time.strftime(Constants.FILENAME_DATEFORMAT)}_original.mp4',
                        self.fourcc,
                        current_fps,
                        (original_frame.shape[1],
                         original_frame.shape[0]),
                    1)
                    self.marked_recorder = cv2.VideoWriter(
                        f'./app/data/{self.last_object_detected_time.strftime(Constants.FILENAME_DATEFORMAT)}_marked.mp4',
                        self.fourcc,
                        current_fps,
                        (frame.shape[1],
                         frame.shape[0]),
                    1)
                self.boxes = boxes
                self.confidences = confidences
                self.class_names = class_names
                self.last_object_detected_time = datetime.now()
                self.recording_status_label.text = Constants.OBJECT_RECORDING
                print('detecting objects')
            # Only check if still recording
            elif self.recording:
                if self._seconds_since_last_detection() > Constants.IDLE_SECONDS:
                    self._stop_recording()
                    self.recording_status_label.text = Constants.NO_OBJECT_DETECTED
                    print('not detecting objects, going idle')
                    # Clean up old detections
                    self.boxes = []
                    self.class_names = []
                    self.confidences = []
                else:
                    print('not detecting objects')

            for box, confidence, class_name in zip(self.boxes, self.confidences, self.class_names):
                x, y, w, h = box
                color = Constants.CLASS_COLORS_TO_USE[class_name]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                cv2.putText(frame,
                            f'{class_name} ({confidence:.1%})',
                            (x, y + 30),
                            self.font,
                            2,
                            (255, 255, 255, 0),
                            2,
                            lineType=cv2.LINE_AA)

            opacity = max(1 - (self._seconds_since_last_detection() / Constants.IDLE_SECONDS), 0)
            # We show the bounding boxes with different opacity to indicate the time since last detection
            # Most opaque => most recent detection
            frame = cv2.addWeighted(frame, opacity, original_frame, 1 - opacity, gamma=0)

        if self.recording:
            self.marked_recorder.write(frame)
            print('add frame')
            self.original_recorder.write(original_frame)
        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Recording function to capture objects if present
    def start_stop_recording(self, *args):
        if self.start_stop_recording_button.icon == Constants.START_RECORDING:
            self.should_detect = True
            self.start_stop_recording_button.icon = Constants.STOP_RECORDING
            self.recording_status_label.text = Constants.NO_OBJECT_DETECTED
        elif self.start_stop_recording_button.icon == Constants.STOP_RECORDING:
            self.should_detect = False
            if self.recording:
                self._stop_recording()
            self.start_stop_recording_button.icon = Constants.START_RECORDING
            self.recording_status_label.text = Constants.NOT_RECORDING

    def _create_layout(self):
        # Loading camera
        self.capture = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.starting_time = datetime.now()
        self.frame_id = 0
        self.last_object_detected_time = datetime.now()
        Clock.schedule_interval(self.update, timeout=(1.0/Constants.MIN_FPS))

        # Main layout components 
        self.should_detect = False
        self.recording = False
        self.web_cam = Image(
            size=Window.size,
            size_hint=(1, 1)
        )
        self.start_stop_recording_button = MDIconButton(
            icon=Constants.START_RECORDING,
            icon_color=(1, 0, 0, 1),
            md_bg_color=(1, 1, 1, 1),
            icon_size='40sp',
            on_press=self.start_stop_recording
        )
        self.recording_status_label = MDLabel(
            text=Constants.NOT_RECORDING,
            size_hint=(1, .1),
            halign="center",
            font_style="H5"
        )

        base_screen = MDScreen()
        base_screen.add_widget(self.web_cam)
        top_center = AnchorLayout(anchor_x='center', anchor_y='top', padding=[10, 10, 10, 10])
        top_center.add_widget(self.recording_status_label)
        base_screen.add_widget(top_center)
        side_layout = AnchorLayout(anchor_x='right', anchor_y='center', padding=[10, 10, 10, 10])
        side_layout.add_widget(self.start_stop_recording_button)
        base_screen.add_widget(side_layout)
        return base_screen

    def _load_model(self):
        self.yolo_v3 = YoloV3()

    def _seconds_since_last_detection(self):
        return (datetime.now() - self.last_object_detected_time).total_seconds()

    def _stop_recording(self):
        self.recording = False
        self.original_recorder.release()
        self.original_recorder = None
        self.marked_recorder.release()
        self.marked_recorder = None


if __name__ == '__main__':
    WildlifeCameraTrapApp().run()
