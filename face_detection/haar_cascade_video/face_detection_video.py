import cv2

from argparse import ArgumentParser


# trích xuất đường dẫn video (tham số khi chạy code)
parser = ArgumentParser(description="Apply Haar Cascade model on videos")
parser.add_argument('--video', dest="video_path", help='Path to video')
parser.add_argument('--webcam', dest="webcam_number", help='Webcam number', type=int)
args = parser.parse_args()
video_path = args.video_path or args.webcam_number
assert video_path is not None, "Please provide either video path (--image) parameter or webcam number (--webcam)"

# load mô hình nhận diện khuôn mặt
model_path = "models/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(model_path)

# kết nối với video cần được xử lý
video = cv2.VideoCapture(video_path)

while(video.isOpened()):
    # đọc từng khung hình
    ret, frame = video.read()
    frame = cv2.resize(frame, dsize=(360, 240))
    # áp dụng nhận diện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)
    
    face_color = (0, 255, 0)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=face_color, thickness=2)
    
    # hiển thị từng khung hình đã được xử lý
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ngừng kết nối với video và dừng các khung hình được hiển thị
video.release()
cv2.destroyAllWindows()