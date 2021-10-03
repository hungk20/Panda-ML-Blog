from argparse import ArgumentParser

from utils import (
    compute_face_distances,
    face_image_to_encoding,
    get_most_similar_face,
    load_image,
)

# trích xuất đường dẫn ảnh
parser = ArgumentParser(description="Recognize a face")
parser.add_argument(
    "--image", dest="image_path", help="Path to the image", required=True
)
args = parser.parse_args()
new_image_path = args.image_path
# trích xuất face embeddings / encodings của ảnh khuôn mặt đã biết
known_faces = [
    ("ronaldinho", "images/ronaldinho/ronaldinho1.jpeg"),
    ("ronaldinho", "images/ronaldinho/ronaldinho2.jpeg"),
    ("zidane", "images/zidane/zidane1.jpeg"),
    ("zidane", "images/zidane/zidane2.jpeg"),
]

known_names = []
known_encodings = []
for name, image_path in known_faces:
    image = load_image(image_path)
    encoding = face_image_to_encoding(image)
    known_names.append(name)
    known_encodings.append(encoding)
# trích xuất face embeddings / encodings của ảnh mới
new_image = load_image(new_image_path)
new_encoding = face_image_to_encoding(new_image)
# tính khoảng cách từ embeddings của ảnh mới đến tất cả embeddings 
# của ảnh đã biết và tìm ảnh có khoảng cách gần nhất
face_distances = compute_face_distances(known_encodings, new_encoding)
name, distance = get_most_similar_face(face_distances, known_names)
# xác định tên trong ảnh mới
if distance > 0.6:
    print("Unknown face")
else:
    print("New image was recognized as: {}".format(name))
