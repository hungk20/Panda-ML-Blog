from argparse import ArgumentParser

import cv2

from utils import CLASSES, COLORS, CONFIDENCE, dnn_detection_to_points

# parse the script parameters
parser = ArgumentParser(description="Recognize a object in an image")
parser.add_argument(
    "--image", dest="image_path", help="Path to the image", required=True
)
args = parser.parse_args()
image_path = args.image_path

# load image
image = cv2.imread(image_path)
height, width = image.shape[:2]

# load model
weights_path = "models/MobileNetSSD_deploy.caffemodel"
net_structure_path = "models/MobileNetSSD_deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(net_structure_path, weights_path)

# resize image
input_size = (300, 300)
resized_image = cv2.resize(image, dsize=input_size)

# run model
blob = cv2.dnn.blobFromImage(
    resized_image, scalefactor=0.007843, size=input_size, mean=127.5
)
net.setInput(blob)
detections = net.forward()

# plot bounding-box
for i in range(detections.shape[2]):
    detection = detections[0, 0, i]
    confidence = detection[2]
    # only keep strong detections
    if confidence > CONFIDENCE:
        idx = int(detection[1])  # class index
        x1, y1, x2, y2 = dnn_detection_to_points(detection, width, height)
        class_label = CLASSES[idx] + ":" + str(round(confidence, 2))
        class_color = COLORS[idx]
        cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=class_color, thickness=2)
        cv2.putText(
            image,
            class_label,
            org=(x1 + 5, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=class_color,
        )

# show the output frame
cv2.imshow("Output Image", image)
cv2.waitKey(0)  # press any key to stop
