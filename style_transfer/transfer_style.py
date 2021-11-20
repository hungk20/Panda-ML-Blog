from argparse import ArgumentParser

import cv2

from utils import MEAN_SUBTRACTIONS, post_process_neural_style_image

# parse the script parameters
parser = ArgumentParser()
parser.add_argument("--image", dest="image_path", required=True)
parser.add_argument("--model", dest="model_path", required=True)
args = parser.parse_args()
image_path = args.image_path
model_path = args.model_path
# load image
image = cv2.imread(image_path)
height, width = image.shape[:2]

# load model
net = cv2.dnn.readNetFromTorch(model_path)

# run model
blob = cv2.dnn.blobFromImage(
    image, 1.0, (width, height), MEAN_SUBTRACTIONS, swapRB=False, crop=False
)
net.setInput(blob)
output = net.forward()

# post process the output result
output = post_process_neural_style_image(output)

# display image
cv2.imshow("Image", image)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
