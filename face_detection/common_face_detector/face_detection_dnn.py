import cv2
from argparse import ArgumentParser

from utils import dnn_detection_to_points


# trích xuất đường dẫn ảnh (tham số khi chạy code)
parser = ArgumentParser(description="Apply Haar Cascade model on images")
parser.add_argument('--image', dest="image_path", help='Path to image', required=True)
args = parser.parse_args()
image_path = args.image_path

# load ảnh và chuyển sang ảnh RGB
# (mô hình cần ảnh đen trắng đầu vào)
image = cv2.imread(image_path)
height, width = image.shape[:2]
print("Load ảnh: {}".format(image_path))
print("Cỡ ảnh: {}".format(image.shape))

# load mô hình nhận diện khuôn mặt
model_path = "models/opencv_face_detector_uint8.pb"
config_path = "models/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# thay đổi độ phân giải của ảnh cho phù hợp với mô hình
input_size = (300, 300)
resized_image = cv2.resize(image, dsize=input_size)
# nhận diện khuôn mặt trong ảnh
blob = cv2.dnn.blobFromImage(resized_image, scalefactor=1.0, size=input_size, mean=[123, 117, 104])
net.setInput(blob)
detections = net.forward()

green_color = (0, 255, 0)
for i in range(detections.shape[2]):
    detection = detections[0, 0, i]
    confidence = detection[2]
    if confidence > 0.4:
        x1, y1, x2, y2 = dnn_detection_to_points(detection, width, height)
        cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=green_color, thickness=2)

# Hiển thị ảnh
cv2.imshow("Output Image with Face Detection", image)
cv2.waitKey(0)
