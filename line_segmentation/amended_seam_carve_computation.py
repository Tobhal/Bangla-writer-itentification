import cv2
import numpy as np
import itertools

from line_info import LineInfo_Compleat
from types import Image, Image_data


def advanced_generation_seam_curve_image_with_binary_info(
        image_derivative: Image,
        length: int,
        width: int,
        binary,
        traverse_pointer: LineInfo_Compleat
) -> Image:
    """
    c++ file: AmendedSeamCarveComputation.cpp
    func: int **Advanced_Generation_SeamCurveImagewithbinaryinfo
    """
    seam_derivative_image: Image = np.zeros((length, width), np.uint8)
    seam_image_width, seam_image_height, seam_image_channels = seam_derivative_image.shape
    seam_image_data: Image_data = np.zeros((length * width), np.uint8)

    transpose_derivative_image: Image = np.zeros((length, width), np.uint8)
    transpose_image_width, transpose_image_height, transpose_image_channels = transpose_derivative_image.shape
    transpose_image_data: Image_data = np.zeros((length * width), np.uint8)

    image_seam_curve: Image = np.zeros((length, width), np.uint8)
    seam = np.zeros((length, width), np.uint8)
    transpose_seam: Image = np.zeros((width, length), np.uint8)

    temp_width, temp_height, temp_channels = transpose_derivative_image.shape
    temp_step = 1
    temp_seam_data: Image_data = np.zeros((length * width), np.uint8)

    seam_data: Image_data = np.zeros((length * width), np.uint8)

    flag_for_no_purpose = 0
    max_gray_val = 0

    for i, j in itertools.product(range(length), range(width)):
        temp_seam_data[i * temp_step + j * temp_channels] = image_derivative[i][j]

        max_gray_val = image_derivative[i][j] if image_derivative[i][j] < max_gray_val else max_gray_val

    line_binary: Image = np.zeros((length, width), np.uint8)

    for i, j in itertools.product(range(width), range(length)):
        line_binary[i][j] = 125 if traverse_pointer.line_binary[i][j] in (25, 50) else 255

    for i, j in itertools.product(range(length), range(width)):
        image_derivative[i][j] = max_gray_val if binary[i][j] >= 100 else image_derivative[i][j]

        temp_seam_data[i * temp_step + j * temp_channels] = image_derivative[i][j]

        image_seam_curve[i][j] = image_derivative[i][j] if i == 0 else 0

    min_of_neighbor = 0

    left_margin = 0
    right_margin = width - 1

    for i in range(width):
        if traverse_pointer.line_binary[i][width - 1] == 50:
            left_margin = i
            break

    for i in range(width - 1, 0, -1):
        if traverse_pointer.line_binary[i][width - 1] == 25:
            right_margin = i
            break

    primal_column = 0

    for i in range(1, length):
        previous_left_margin = left_margin
        previous_right_margin = right_margin
        left_margin = 0
        right_margin = width - 1

        for j in range(width):
            if traverse_pointer.line_binary[j][i] == 100:
                primal_column = j
                break

        for j in range(width):
            if traverse_pointer.line_binary[j][i] == 50:
                left_margin = j
                break

        for j in range(width - 1, 0, -1):
            if traverse_pointer.line_binary[j][i] == 25:
                right_margin = j
                break

        for j in range(left_margin, right_margin + 1):
            left = max(j - 1, previous_left_margin)
            right = min(j + 1, previous_right_margin)
            min_of_neighbor = 50_000

            bin_flag = 0

            for k in range(left, right):
                min_of_neighbor = image_seam_curve[j - 1][k] if image_seam_curve[j - 1][k] else min_of_neighbor

            image_seam_curve[i][j] = abs(image_derivative[i][j] + min_of_neighbor)

    print('start back tracking')

    min_index_j = 0
    left_margin = 0
    right_margin = width - 1

    for i in range(width):
        if traverse_pointer.line_binary[i][length - 1] == 50:
            left_margin = i
            break

    for i in range(width - 1, 0, -1):
        if traverse_pointer.line_binary[i][length - 1] == 25:
            right_margin = i
            break

    min_of_neighbor = 50_000_000
    previous_min_index = 1

    for i in range(left_margin, right_margin):
        if min_of_neighbor > image_seam_curve[length - 1][i]:
            min_of_neighbor = image_seam_curve[length - 1][i]
            previous_min_index = i

    seam[length - 1][previous_min_index] = 255
    previous_min_index_j = 0

    for i in range(length - 2, 0, -1):
        previous_left_margin = left_margin + 1
        previous_right_margin = right_margin - 1
        left_margin = 0
        right_margin = width - 1

        primal_column = 0
        # The purpose of the next for loops is to sett left, right and plausible column.

        for j in range(1, width):
            if traverse_pointer.line_binary[j][i] == 100:
                primal_column = j
                break

        for j in range(width):
            if traverse_pointer.line_binary[j][i] == 50:
                left_margin = j
                break

        for j in range(width - 1, 0, -1):
            if traverse_pointer.line_binary[j][i] == 25:
                right_margin = j
                break

        min_of_neighbor = 50_000_000

        left = max(previous_min_index - 1, left_margin + 1)
        right = min(previous_min_index + 1, right_margin - 1)

        for j in range(left, right):
            if min_of_neighbor > image_seam_curve[i][j]:
                min_of_neighbor = image_seam_curve[i][j]
                min_index_j = j

        previous_min_index_j = min_index_j

        seam[i][min_index_j] = 255

    for i, j in itertools.product(range(width), range(length)):
        transpose_seam[i][j] = seam[j][i]

    for i, j in itertools.product(range(width), range(length)):
        seam_data[i * temp_step + j * temp_channels] = transpose_seam[i][j]

    cv2.imwrite('only-seam-part.jpg', seam_derivative_image)

    return transpose_seam




