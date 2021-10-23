from argparse import ArgumentParser

import cv2

from utils import (
    CLASSES,
    COLORS,
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    draw_bounding_box_with_label,
    yolo_box_to_points,
)

# parse the script parameters
parser = ArgumentParser(description="Recognize objects in an image")
parser.add_argument(
    "--image", dest="image_path", help="Path to the image", required=True
)
args = parser.parse_args()
image_path = args.image_path

# load image
image = cv2.imread(image_path)
height, width = image.shape[:2]

# load model
weights_path = "models/yolov4.weights"
config_path = "models/yolov4.cfg"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# run model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
class_ids, scores, boxes = model.detect(image, confThreshold=0.6, nmsThreshold=0.4)

# draw bounding box for each object
for (class_id, score, box) in zip(class_ids, scores, boxes):
    label = "%s: %.2f" % (CLASSES[class_id[0]], score)
    color = COLORS[class_id[0]]
    x1, y1, x2, y2 = yolo_box_to_points(box)
    draw_bounding_box_with_label(image, x1, y1, x2, y2, label, color)


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
