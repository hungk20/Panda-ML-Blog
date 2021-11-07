import cv2
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
np.random.seed(111)  # set seed to ensure color is consistent
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# confidence level
CONFIDENCE = 0.5


def dnn_detection_to_points(detection, width, height):
    x1 = int(detection[3] * width)
    y1 = int(detection[4] * height)
    x2 = int(detection[5] * width)
    y2 = int(detection[6] * height)

    return x1, y1, x2, y2


def draw_bounding_box_with_label(image, x1, y1, x2, y2, label, color, thickness=2):
    """Helper function to draw a bounding box with class label

    Parameters
    ----------
    image : np.ndarray
        Image object read by cv2
    x1, y1, x2, y2 : float
        Coordinates of the bounding box (top left) to (bottom right)
    label : str
        Text to be shown in bounding box, usually classname
    color : tuple
        BGR color
    thickness : int, optional
        Thickness of the bounding box
    """
    # draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
    # draw a rectangle that contains label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness=thickness)
    cv2.rectangle(
        image,
        (x1 - int(thickness / 2), y1 - label_size[1]),
        (x1 + label_size[0], y1),
        color,
        cv2.FILLED,
    )
    # draw label
    cv2.putText(
        image, label, (x1, y1), font, font_scale, color=(0, 0, 0)
    )  # black label
