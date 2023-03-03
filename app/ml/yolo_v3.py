import cv2
import numpy as np

from constants import Constants

class YoloV3:

    def __init__(self
                 ):
        self.model = cv2.dnn.readNet(
            "./app/ml/yolov3-tiny.weights",
            "./app/ml/yolov3-tiny.cfg")
        self.classes = []
        with open("./app/ml/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.model.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        self.selected_classes = Constants.COCO_CLASSES_TO_USE.keys()

    def detect_objects(self, frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(self.output_layers)

        class_names = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                class_name = self.classes[class_id]
                if confidence > Constants.CONFIDENCE_THRESHOLD and class_name in self.selected_classes:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_names.append(self.classes[class_id])

        # Filter out only the detections coming from Non-maximum suppression
        indexes = list(cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3))
        boxes = self.__filter_by_indexes(boxes, indexes)
        confidences = self.__filter_by_indexes(confidences, indexes)
        class_names = self.__filter_by_indexes(class_names, indexes)

        return boxes, confidences, class_names

    def __filter_by_indexes(self, array, indexes):
        return [array[index] for index in indexes]
