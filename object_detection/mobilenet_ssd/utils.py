import numpy as np


# 20 class labels in MobileNetSSD model
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
# bounding box color
np.random.seed(1000)  # set seed to ensure color is consistent
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# confidence level
CONFIDENCE = 0.5


def dnn_detection_to_points(detection, width, height):
    x1 = int(detection[3] * width)
    y1 = int(detection[4] * height)
    x2 = int(detection[5] * width)
    y2 = int(detection[6] * height)

    return x1, y1, x2, y2
