import cv2
import numpy as np

from types import Image, Image_data


def image_gaussian_transform(
        image: Image,
        output_src: str
) -> Image:
    """
    c++ file: Process_image_Gaussian.cpp
    func: Image_GaussianTransform
    """
    image_rows, image_columns, image_channels = image.shape

    data: Image_data = image.copy().flatten()
    gausian_data: Image_data = image.copy().flatten()

    pixmax = 0

    return_gaussian_image: Image = np.zeros((image_rows, image_columns, 3), np.uint8)

    gaussian_input: Image_data = np.zeros((image_rows * image_columns), np.double)

    for i in range(image_columns):
        for j in range(image_rows):
            gaussian_input[(i * image_columns) + j] = data[i * 1 + j * image_channels]

    cv2.imwrite(f'{output_src}before_gaussian_smooth.jpg', image)

    gaussian_output = anigauss(gaussian_input, image_columns, image_rows, 2.0, 2.0, 0.0 - 90.0, 0, 0)

    for i in range(image_columns):
        for j in range(image_rows):
            gausian_data[i * 1 + j * image_channels] = gaussian_output[(i * image_rows) + j]
            return_gaussian_image[i][j] = gaussian_output[(i * image_rows) + j]

    cv2.imwrite(f'{output_src}afther_gaussian_smooth.jpg', return_gaussian_image)

    return return_gaussian_image
