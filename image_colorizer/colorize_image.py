from argparse import ArgumentParser

import numpy as np
import cv2

from utils import l_channel_from_lab_image, rgb_from_l_and_ab

# parse the script parameters
parser = ArgumentParser()
parser.add_argument("--image", dest="image_path", required=True)
args = parser.parse_args()
image_path = args.image_path

# load model
weights_path = "models/colorization_release_v2.caffemodel"
config_path = "models/colorization_deploy_v2.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, weights_path)

# add quantized_ab center, will be used for rebalancing
pts = np.load("models/pts_in_hull.npy")
pts = pts.transpose().reshape(2, 313, 1, 1).astype("float32")
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# load image
image = cv2.imread(image_path)
height, width = image.shape[:2]
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# resize, extract L channel, perform mean centering
resized = cv2.resize(lab, (224, 224))
L = l_channel_from_lab_image(resized)
L -= 50

# run model
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# construct colored image from original L channel & predicted ab channel
predicted_ab = cv2.resize(ab, (width, height))  # resize to original image size
original_L = l_channel_from_lab_image(lab)
colorized = rgb_from_l_and_ab(original_L, predicted_ab)

# display image
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
