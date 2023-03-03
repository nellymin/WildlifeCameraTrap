import cv2
from kivy.core.window import Window
from kivymd.uix.screen import MDScreen
import kivy
kivy.require('1.0.6') 
from kivymd.app import MDApp
class WildlifeCameraTrapApp(MDApp):
    def build(self):
        return self._create_layout()
    # Run continuously to get webcam feed
    def update(self, *args):
        _, frame = self.capture.read()
        self.frame_id += 1

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    def _create_layout(self):
        self.capture = cv2.VideoCapture(0)
        self.frame_id = 0
        self.web_cam = Image(
            size=Window.size,
            size_hint=(1, 1)
            )

        base_scren = MDScreen()
        base_scren.add_widget(self.web_cam)
        return base_scren

if __name__ == '__main__':
    WildlifeCameraTrapApp().run()