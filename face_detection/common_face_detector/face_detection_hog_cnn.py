import cv2
import dlib
from argparse import ArgumentParser


def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	# startX = max(0, startX)
	# startY = max(0, startY)
	# endX = min(endX, image.shape[1])
	# endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)


# trích xuất đường dẫn ảnh (tham số khi chạy code)
parser = ArgumentParser(description="Apply Haar Cascade model on images")
parser.add_argument('--image', dest="image_path", help='Path to image', required=True)
args = parser.parse_args()
image_path = args.image_path

# load ảnh và chuyển sang ảnh đen trắng 
# (mô hình cần ảnh đen trắng đầu vào)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Load ảnh: {}".format(image_path))
print("Cỡ ảnh: {}".format(gray.shape))

# load mô hình nhận diện khuôn mặt
hogFaceDetector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

# nhận diện khuôn mặt trong ảnh
faces = hogFaceDetector(gray, 0)

boxes = [convert_and_trim_bb(image, r.rect) for r in faces]

# Vẽ đường bao cho từng khuôn mặt
green_color = (0, 255, 0)
# loop over the bounding boxes
for (x, y, w, h) in boxes:
	# draw the bounding box on our image
	cv2.rectangle(image, (x, y), (x + w, y + h), green_color, thickness=2)

# Hiển thị ảnh
cv2.imshow("Output Image with Face Detection", image)
# cv2.imwrite("images/output_full.jpeg", image)  # cv2.resize(image, (720, 480))
cv2.waitKey(0)
