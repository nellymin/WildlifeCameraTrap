import cv2

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
