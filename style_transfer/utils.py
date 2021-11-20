MEAN_SUBTRACTIONS = [103.939, 116.779, 123.680]


def post_process_neural_style_image(image, mean_subtractions=MEAN_SUBTRACTIONS):
    # output image is in shape (batch, channel, height, width)
    # => reshape to (channel, height, width) as we use batch = 1
    image = image.reshape((3, image.shape[2], image.shape[3]))
    # add back in the mean subtraction
    image[0] += mean_subtractions[0]
    image[1] += mean_subtractions[0]
    image[2] += mean_subtractions[0]
    # scaling
    image /= 255.0
    # then swap the channel ordering
    image = image.transpose(1, 2, 0)  # (3, H, W) => (H, W, 3)

    return image
