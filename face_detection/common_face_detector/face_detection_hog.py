import cv2
import dlib
from argparse import ArgumentParser

from utils import hog_face_to_points


# trích xuất đường dẫn ảnh (tham số khi chạy code)
parser = ArgumentParser(description="Apply Haar Cascade model on images")
parser.add_argument('--image', dest="image_path", help='Path to image', required=True)
args = parser.parse_args()
image_path = args.image_path

# load ảnh và chuyển sang ảnh đen trắng 
# (mô hình cần ảnh đen trắng đầu vào)
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Load ảnh: {}".format(image_path))

# load mô hình nhận diện khuôn mặt
hog_face_detector = dlib.get_frontal_face_detector()

# nhận diện khuôn mặt trong ảnh
faces = hog_face_detector(rgb_image, 0)

# Vẽ đường bao cho từng khuôn mặt
green_color = (0, 255, 0)
for face in faces:
    x1, y1, x2, y2 = hog_face_to_points(face)
    cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=green_color, thickness=2)

# Hiển thị ảnh
cv2.imshow("Output Image with Face Detection", image)
cv2.waitKey(0)
