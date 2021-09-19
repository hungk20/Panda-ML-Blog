def haar_face_to_points(face):
    x1, y1, w, h = face
    x2 = x1 + w 
    y2 = y1 + h

    return x1, y1, x2, y2


def hog_face_to_points(rect):
	x1 = rect.left()
	y1 = rect.top()
	x2 = rect.right()
	y2 = rect.bottom()

	return x1, y1, x2, y2


def dnn_detection_to_points(detection, width, height):
    x1 = int(detection[3] * width)
    y1 = int(detection[4] * height)
    x2 = int(detection[5] * width)
    y2 = int(detection[6] * height)

    return x1, y1, x2, y2
