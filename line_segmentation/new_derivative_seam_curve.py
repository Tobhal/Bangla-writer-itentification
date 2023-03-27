import numpy as np
import itertools

from types import Image, Image_data


def new_generative_derivative_image(
        img: Image,
        length: int,
        width: int
) -> Image:
    """
    c++ file: New_Derivative_SeamCurve.cpp
    func: int **newGenerative_DerivativeImage
    """
    derivative_image: Image = np.zeros((width, length), np.uint8)
    enhance_image: Image = np.zeros((width, length), np.uint8)

    derivative_image_width, derivative_image_height, derivative_image_channels = derivative_image.shape
    derivative_image_step = 1
    derivative_image_data: Image_data = np.zeros((width + length), np.uint8)

    for i, j in itertools.product(range(length), range(width)):
        derivative_image_data[i * derivative_image_step + j * derivative_image_channels] = img[i][j]

    image_derivative: Image = np.zeros((length, width), np.uint8)
    image_derivative_transpose: Image= np.zeros((width, length), np.uint8)

    for i, j in itertools.product(range(length), range(width)):
        if (i > 0) and (i < length - 1) and (j > 0) and (j < width - 1):
            dx = ((img[i][j + 1]) - (img[i][j - 1])) / 2
            dy = ((img[i + 1][j]) - (img[i - 1][j])) / 2
            image_derivative[i][j] = abs(dx + dy)

    for i in range(length):
        image_derivative[i][0] = image_derivative[i][1]
        image_derivative[i][width - 1] = image_derivative[i][width - 2]

    for j in range(width):
        image_derivative[0][j] = image_derivative[1][j]
        image_derivative[length - 1][j] = image_derivative[length - 2][j]

    for i, j in itertools.product(range(length), range(width)):
        derivative_image_data[i * derivative_image_step + j * derivative_image_channels] = image_derivative[i][j]

    for i, j in itertools.product(range(length), range(width)):
        image_derivative_transpose[j][i] = image_derivative[i][j]

    return image_derivative_transpose
