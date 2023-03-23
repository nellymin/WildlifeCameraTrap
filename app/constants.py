class Constants:
    START_RECORDING = 'record'
    STOP_RECORDING = 'stop'
    OBJECT_DETECTED = 'record-rec'
    OBJECT_RECORDING = 'Recording detected object(s)...'
    NO_OBJECT_DETECTED = 'No object(s) detected'
    NOT_RECORDING = ''
    IDLE_SECONDS = 3
    # The confidence at which to return predictions
    CONFIDENCE_THRESHOLD = 0.2
    MIN_FPS = 20
    CLASS_COLORS_TO_USE = {
        # Mapped to the BGR color that will be used for that Class
        'person': (128, 128, 128),
        'bird': (133, 21, 199),
        'cat': (32, 165, 218),
        'dog': (60, 20, 220),
        'horse': (63, 133, 205),
        'sheep': (50, 205, 154),
        'cow': (255, 144, 30),
        'elephant': (144, 128, 112),
        'bear': (19, 69, 139),
        'zebra': (128, 128, 0),
        'giraffe': (0, 165, 255)
    }
    COCO_CLASSES_TO_USE = CLASS_COLORS_TO_USE.keys()
