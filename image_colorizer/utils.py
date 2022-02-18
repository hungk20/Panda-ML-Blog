import cv2
import numpy as np


def l_channel_from_lab_image(lab):
    """Get the lightness channel (L) from a Lab image"""
    return cv2.split(lab)[0]


def rgb_from_l_and_ab(L, ab):
    """Comebine L channel & ab channel to RGB image"""
    rgb = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # convert from Lab to RGB
    rgb = cv2.cvtColor(rgb, cv2.COLOR_LAB2BGR)
    # clip values outside [0 ,1]
    rgb = np.clip(rgb, 0, 1)
    # ensure pixel value to integer
    rgb = (255 * rgb).astype("uint8")

    return rgb
