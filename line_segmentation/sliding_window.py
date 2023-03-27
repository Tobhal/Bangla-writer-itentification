import cv2
import numpy as np

from types import Image, Image_data


def sliding_window(
        image: Image,
        output_src: str
) -> Image:
    """
    c++ file: sliding_window.cpp
    func: IplImage *SlidingWindow
    """
    image_rows, image_columns, image_channels = image.shape

    img: Image = np.zeros((image_rows, image_columns, 1), np.uint8)

    g_img: Image = np.zeros((image_rows, image_columns, 1), np.uint8)
    g_img[:] = (255, 255, 255)

    gaussian_image_big_sigma: Image = cv2.GaussianBlur(image, (21, 21), sigmaX=5, sigmaY=5)
    gaussian_image_small_sigma: Image = cv2.GaussianBlur(image, (21, 21), sigmaX=3, sigmaY=3)

    gaussian_image_subtract: Image = cv2.subtract(gaussian_image_big_sigma, gaussian_image_small_sigma)

    cv2.imwrite(f'{output_src}big_sigma_gaussian_image.png', gaussian_image_big_sigma)
    cv2.imwrite(f'{output_src}small_sigma gaussian_image.png', gaussian_image_small_sigma)
    cv2.imwrite(f'{output_src}subtract_gaussian_image.png', gaussian_image_subtract)

    img_hist_equalized: Image = cv2.equalizeHist(gaussian_image_subtract)

    for i in range(image_rows):
        for j in range(image_columns):
            img[i][j] = img_hist_equalized[i][j]

    cv2.imwrite(f'{output_src}contrast_enhanced_derivative.png', img_hist_equalized)

    """
    cv2.imwrite(f'{output_src}before_derivative.png', img)
    img = derative_image(img, image_rows, image_columns, (5, 5))
    
    for i in range(image_rows):
        for j in range(image_columns):
            img[i][j] = img_hist_equalized[i][j]
            
    img_hist_equalized = cv2.GAussianBlur(image, (5, 5), sigmaX=1, sigmaY=1)
    cv2.imwrite(f'{output_src}after_derivative.png', img)
    """

    ipltemp1: Image = img_hist_equalized.copy()

    layout: Image = np.zeros((image_rows, image_columns, 1), dtype=np.char)

    horizontal_projection: Image = np.zeros((image_rows, 1, 1), dtype="CV_32SC1")
    vertical_projection: Image = np.zeros((image_columns, 1, 1), dtype="CV_32SC1")

    horizontal_projection_layout: Image = np.zeros((image_rows, image_columns * 2, 1), dtype="CV_32SC1")
    vertical_projection_layout: Image = np.zeros((image_rows * 2, image_columns, 1), dtype="CV_32SC1")

    horizontal_projection_layout[:] = 255
    vertical_projection_layout[:] = 255

    for i in range(image_rows):  # Is layout in c++ code
        for j in range(image_columns):  # Is layout in c++ code
            layout[i][j] = 0 if layout[i][j] >= 225 else 255

    # perform vertical and horizontal smoothing here
    for i in range(image_rows):  # Is layout in c++ code
        for j in range(1, image_columns - 1):  # Is layout in c++ code
            pixel_val = layout[i][j - 1]

            if (pixel_val == 0) and (layout[i][j] == 255):
                blank_space, inner_j = 0, 0

                for inner_j in range(j, image_columns):
                    if layout[i][inner_j] == 255:
                        blank_space += 1
                    else:
                        break

                    if blank_space <= image_columns/100*2:  # Is layout in c++ code
                        cv2.line(layout, (j, i), (inner_j, i), 0)

                    inner_j += 1

    for i in range(1, image_columns - 1):   # Is layout in c++ code
        for j in range(1, image_rows):  # Is layout in c++ code
            pixel_val = layout[i - 1][j]

            if (pixel_val == 0) and (layout[i][j] == 255):
                blank_space, inner_i = 0, 0

                for inner_i in range(i, image_rows):
                    if layout[inner_i][j] == 255:
                        blank_space += 1
                    else:
                        break

                    if blank_space <= image_rows/100*2:  # Is layout in c++ code
                        cv2.line(layout, (j, i), (j, inner_i), 0)

    cv2.imwrite(f'{output_src}smoothed.jpg', layout)

    # Horizontal
    total_horizontal_projection = 0

    for i in range(image_rows):     # Is layout in c++ code
        horizontal_projection_single = 0

        for j in range(image_columns):  # Is layout in c++ code
            horizontal_projection_single += 1 if layout[i][j] == 0 else 0

        horizontal_projection[i][0] = horizontal_projection_single
        total_horizontal_projection += horizontal_projection_single
        cv2.line(
            horizontal_projection_layout,
            (image_columns, i),     # Is layout in c++ code
            (image_columns + horizontal_projection_single, i),  # Is layout in c++ code
            125
        )

    avg_horizontal_projection = total_horizontal_projection / image_rows    # Is layout in c++ code
    cv2.line(
        horizontal_projection_layout,
        (image_columns + (avg_horizontal_projection/2), 1),     # Is layout in c++ code
        (image_columns + (avg_horizontal_projection/2), image_rows),    # Is layout in c++ code
        10, 20
    )

    cv2.imwrite(f'{output_src}horizontal_projection.jpg', horizontal_projection)

    # Vertical
    total_vertical_projection = 0

    for i in range(image_columns):  # Is layout in c++ code
        vertical_projection_single = 0

        for j in range(image_rows):     # Is layout in c++ code
            vertical_projection_single += 1 if layout[i][j] == 0 else 0

        vertical_projection[i][0] = vertical_projection_single
        total_vertical_projection += vertical_projection_single
        cv2.line(
            vertical_projection_layout,
            (j, i),
            (i, j + vertical_projection_single),
            125
        )

    avg_vertical_projection = total_vertical_projection / image_columns     # Is layout in c++ code
    cv2.line(
        vertical_projection,
        (0, image_rows + (avg_vertical_projection/2)),  # Is layout in c++ code
        (image_columns, image_rows + (avg_vertical_projection/2)),  # Is layout in c++ code
        0, 20
    )

    cv2.imwrite(f'{output_src}vertical_projection.jpg', vertical_projection_layout)

    # check left column
    first_left_text_col = image_rows
    for i in range(image_columns/2):    # Is layout in c++ code
        count = 0

        if vertical_projection[i][0] < (avg_vertical_projection / 2):

            for inner_i in range(i, image_columns):     # Is layout in c++ code
                if vertical_projection[inner_i][0] < (avg_vertical_projection / 2):
                    count += 1
                else:
                    break

                if count >= (image_columns / 100 * 2):
                    first_left_text_col = l + count / 2
                    break

    # check right column
    last_right_text_col = image_columns
    for i in range(image_columns - 1, image_columns / 2, -1):
        count = 0

        if vertical_projection[i][0] < (avg_vertical_projection / 2):
            count = 0

            for inner_i in range(i, 0, -1):
                if vertical_projection[inner_i][0] < (avg_vertical_projection / 2):
                    count += 1
                else:
                    break

                if count > 0:
                    last_right_text_col = i - count / 2

    # check top row
    first_top_text_row = 0
    for i in range(image_rows):
        count = 0

        if horizontal_projection[i][0] < (avg_horizontal_projection / 2):
            count = 0
            for inner_i in range(i, image_columns):
                if horizontal_projection[inner_i][0] < (avg_horizontal_projection / 2):
                    count += 1
                else:
                    break

                if count >= image_rows / 100 * 2:
                    first_top_text_row = i + count / 2

    # check bottom row
    last_bottom_text_row = 0
    for i in range(image_rows, 0, -1):
        count = 0

        if horizontal_projection[i][0] < (avg_horizontal_projection / 2):
            count = 0

            for inner_i in range(i, 0, -1):
                if horizontal_projection[inner_i][0] < (avg_horizontal_projection / 2):
                    count += 1
                else:
                    break

                if count >= image_rows / 100 * 2:
                    last_bottom_text_row = i - count / 2
                    break

    return ipltemp1
