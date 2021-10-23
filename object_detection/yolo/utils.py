import cv2
import numpy as np

class_names_path = "models/coco.names"
with open(class_names_path, "r") as f:
    CLASSES = f.read().splitlines()

# bounding box color
np.random.seed(1000)  # set seed to ensure color is consistent
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# confidence level
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


def yolo_box_to_points(box):
    """Helper function to convert yolo box result to coordinates,
    make it easier to draw in openCV

    Parameters
    ----------
    box : list[float]
        Bounding box location

    Returns
    -------
    floats
        Coordinates in order (top left) to (bottom right)
    """
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
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
