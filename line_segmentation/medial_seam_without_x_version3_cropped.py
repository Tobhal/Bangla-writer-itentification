import cv2
import numpy as np

from line_info import LineInfo_Compleat

from types import Image


def update_compute_medial_seams_cropped(
        surf_point_img: Image,
        gray_image: Image,
        s: int,
        smooth: float,
        off: int,
        mark: Image
) -> Image:
    """
    c++ file: MedialSeamWithoutXVersion3_Cropped.cpp
    func: void update_compute_medial_seams_cropped
    """
    start_pointer_compleat_info = LineInfo_Compleat(0, 0, 0, 0, [], None)

    surf_point_img_h, surf_point_img_w, surf_point_img_channels = surf_point_img.shape
    gray_image_h, gray_image_w, gray_image_channels = gray_image.shape

    mat_img: Image = surf_point_img.copy()
    mat_img_crop: Image = surf_point_img.copy()
    original_gray_mat_image: Image = gray_image.copy()
    original_colour_mat_image: Image = np.zeros(())
    duplicate_gray_image: Image = gray_image.copy()
    duplicate_gray_mat_image: Image = duplicate_gray_image.copy()

    cv2.cvtColor(original_gray_mat_image, original_colour_mat_image, cv2.COLOR_GRAY2BGR)

    mat_img_crop_columns, mat_img_crop_rows, mat_img_crop_channels = mat_img_crop.shape

    w = np.floor(mat_img_crop_columns / s)

    bin_img: Image = np.zeros((mat_img_crop_rows, mat_img_crop_columns), np.uint8)
    bin_img_2: Image = np.zeros((mat_img_crop_rows, mat_img_crop_columns), np.uint8)
    bin_output: Image = np.zeros((mat_img_crop_rows, mat_img_crop_columns), np.uint8)

    local_max = []

    k = 0
    for i in range(s):
        horizontal_projection = np.zeros((mat_img_crop_rows, 1), np.int8)

        for j in range(mat_img_crop_rows):
            horizontal_projection_single = 0

            for l in range(k, k + w):
                pixel_val = mat_img_crop[j][l]

                if pixel_val >= 190:
                    horizontal_projection_single += pixel_val

            horizontal_projection[j][0] = horizontal_projection_single

        k += w

        horizontal_projection_rows, horizontal_projection_columns = horizontal_projection.size
        horizontal_projection_smooth = np.zeros((horizontal_projection_rows, horizontal_projection_columns), np.float64)







    return surf_point_img   # Temp return


