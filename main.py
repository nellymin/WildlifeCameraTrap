import cv2
import os

from kivy.graphics import RoundedRectangle
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.scatterlayout import ScatterLayout
from kivymd.theming import Animation
from kivy.core.window import Window
from kivymd.uix.screen import MDScreen
from app.ml.yolo_v3 import YoloV3
from datetime import datetime

import kivy
kivy.require('1.0.6') 
from kivymd.app import MDApp
from kivy.clock import Clock
from kivymd.uix.button.button import MDIconButton
from kivymd.uix.label.label import MDIcon, MDLabel
from kivy.uix.image import Image
from kivy.uix.anchorlayout import AnchorLayout
from kivy.graphics.texture import Texture
from constants import Constants

class WildlifeCameraTrapApp(MDApp):
    def build(self):
        self.boxes = []
        self.class_names = []
        self.confidences = []
        self._load_model()
        return self._create_layout()
    # Run continuously to get webcam feed
    def update(self, *args):
        _, frame = self.capture.read()
        self.frame_id += 1
        if self.should_detect:
            boxes, confidences, class_names = self.yolo_v3.detect_objects(frame)
            # when objects are detected
            if len(boxes):
                self.recording = True
                self.boxes = boxes
                self.confidences = confidences
                self.class_names = class_names
                self.last_object_detected_time = datetime.now()
                # self.last_boxes_detected = boxes
                # self.last_confidences_detected = confidences
                self.recording_status_label.text = Constants.OBJECT_RECORDING
                print('detecting objects')
            # Only check if still recording
            elif self.recording:
                if self._seconds_since_last_detection() > Constants.IDLE_SECONDS:
                    self.recording = False
                    self.recording_status_label.text = Constants.NO_OBJECT_DETECTED
                    print('not detecting objects, going idle')
                    # Clean up old detections
                    self.boxes = []
                    self.class_names = []
                    self.confidences = []
                else:
                    print('not detecting objects')

            frame_copy = frame.copy()
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
                            self._get_gbr('white'),
                            2,
                            lineType=cv2.LINE_AA)
            opacity = max(1 - (self._seconds_since_last_detection() / Constants.IDLE_SECONDS), 0)
            # We show the bounding boxes with different opacity to indicate the time since last detection
            # Most opaque => most recent detection
            frame = cv2.addWeighted(frame, opacity, frame_copy, 1 - opacity, gamma=0)

        elapsed_time = (datetime.now() - self.starting_time).total_seconds()

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
            self.start_stop_recording_button.icon = Constants.START_RECORDING
            self.recording_status_label.text = Constants.NOT_RECORDING

    def _create_layout(self):
        # Loading camera
        self.capture = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.starting_time = datetime.now()
        self.frame_id = 0
        self.last_object_detected_time = datetime.now()
        Clock.schedule_interval(self.update, 1.0/Constants.MAX_FPS)

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
            # text=Constants.START_RECORDING, 
            on_press=self.start_stop_recording
            # size_hint=(.05,.05)
            )
        self.recording_status_label = MDLabel(
            text=Constants.NOT_RECORDING,
            size_hint=(1,.1),
            halign="center",
            font_style="H5"
            )


        base_scren = MDScreen()
        base_scren.add_widget(self.web_cam)
        top_center = AnchorLayout(anchor_x='center', anchor_y='top', padding=[10,10,10,10])
        top_center.add_widget(self.recording_status_label)
        base_scren.add_widget(top_center)
        side_layout = AnchorLayout(anchor_x='right', anchor_y='center', padding=[10,10,10,10])
        side_layout.add_widget(self.start_stop_recording_button)
        base_scren.add_widget(side_layout)
        return base_scren


    def _load_model(self):
        self.yolo_v3 = YoloV3()

    def _seconds_since_last_detection(self):
        return (datetime.now() - self.last_object_detected_time).total_seconds()

if __name__ == '__main__':
    WildlifeCameraTrapApp().run()