import cv2
import numpy as np
import os

from sliding_window import sliding_window
from generate_line_image import generate_line_image
from process_image_gaussian import image_gaussian_transform
from new_derivative_seam_curve import new_generative_derivative_image
from medial_seam_without_x_version3_cropped import update_compute_medial_seams_cropped
from amended_seam_carve_computation import advanced_generation_seam_curve_image_with_binary_info

from line_info import *

from types import Image, Image_data

fp_xml = open("AUB_boundingBox.xml", "a")


def main(file: str):
    output_src = '../output/'
    input_src = '../input/'

    fp_xml.write('<?xml version="1.0"?>\n')
    fp_xml.write('<DocumentList>\n')
    fp_xml.write('<SinglePage FileName="%s">\n')

    for file_name in os.listdir(input_src):
        file_name_with_out_extension = file_name.split('.')[0]

        file_directory = os.path.join(input_src, file_name_with_out_extension)

        raw_img: Image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        raw_img_h, raw_img_w, raw_img_channels = raw_img.shape
        raw_image_shape: Image_data = raw_img.shape  # For reconstructing image shape later

        raw_img_steps = 1

        img: Image = np.zeros((raw_img_h, raw_img_w, 3), np.uint8)
        gray_img: Image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        input_furiour: Image = np.zeros((raw_img_h, raw_img_w, 1), np.uint8)
        ipltemp: Image = raw_img.copy()

        gray_img_2: Image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        seam_gray_img: Image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        only_lines: Image = np.zeros((raw_img_h, raw_img_w, 1), np.uint8)

        g_img: Image = sliding_window(gray_img, output_src)
        g_img_data: Image_data = raw_img.ravel()

        g_img_row, g_img_column, g_image_channels = g_img.shape

        mark: Image = np.zeros((2, g_img_column, 1), np.int16)

        cv2.imwrite(f'{output_src}smooth.jpg', g_img)

        gaussian_image: Image = image_gaussian_transform(g_img, output_src)    # TODO: Implement sub func

        gray_img: Image = update_compute_medial_seams_cropped(g_img, gray_img, 4, 0.000403, 15, mark)  # TODO: Implement func

        # convert 1D gray image to 2D gray image. Probably does not work
        mat_of_g_img: Image = np.zeros((raw_img_h, raw_img_w, 1), np.uint8)
        mat_of_g_img[:, :, 0] = g_img

        travers_pointer = LineInfo_Compleat(0, 0, 0, 0, [], None)
        head = LineInfo_Compleat(0, 0, 0, 0, [], None)

        gray_data: Image_data = np.zeros((raw_img_h * raw_img_w * 3), np.char)

        width, step, channels = raw_img_w, raw_img_steps, raw_img_channels

        # Related to seam image
        seam_width, seam_step, seam_channels = raw_img_w, raw_img_steps, raw_img_channels  # c++: from seam image
        only_seam_data: Image_data = np.zeros((raw_img_h * raw_img_steps + raw_img_w * raw_img_channels),
                                  np.char)  # c++: from seam image

        line_number = 0
        counter_of_lines = 0
        line_number_string = ""
        line_number_char = ''

        travers_pointer = head

        # This is probaly wrong...
        seam_curve_horizontal: Image = np.zeros((raw_img_h, raw_img_w), np.uint8)

        # Also this?
        seam_curve_dist: Image = raw_img.copy()

        final_seam: Image = np.zeros((raw_img_h, raw_img_w), np.uint8)

        medial_seam_first = 0
        medial_seam_last = 0

        original_color_mat_image: Image = np.zeros((raw_img_h, raw_img_w, raw_img_channels), np.uint8)

        only_seam_image: Image = np.zeros((raw_img_h, raw_img_w))

        fp_xml.write(file_name)

        while travers_pointer.next is not None:
            start_row = travers_pointer.start_row
            end_row = travers_pointer.end_row

            start_col = travers_pointer.start_col
            end_col = travers_pointer.end_col

            counter_of_lines += 1
            line_number += 1

            med_seam_image: Image = np.zeros((end_row - start_row + 1, end_col - start_col + 1), np.int16)
            med_seam_gaussian_image: Image = np.zeros((end_row - start_row + 1, end_col - start_col + 1), np.int16)
            med_seam_region: Image = np.zeros((end_row - start_row + 1, end_col - start_col + 1), np.int16)
            bin_line_transpose: Image = np.zeros((end_col - start_col + 1, end_row - start_row + 1), np.int16)

            between_med_seam_region_grayscale: Image = np.zeros(((end_row - start_row) + 1, (end_col - start_col) + 1),
                                                         np.uint8)
            between_med_seam_region_grayscale[:] = 256

            between_med_seam_region_binary: Image = np.zeros(((end_row - start_row) + 1, (end_col - start_col) + 1), np.uint8)
            between_med_seam_region_binary[:] = 256

            only_seam_data[:] = 255

            for i in range(start_row, end_row):
                loop_i = 0

                j, loop_j = 0, start_col
                while loop_j < end_col:
                    med_seam_image[loop_i][j] = gray_data[i * step + loop_j * channels]
                    med_seam_gaussian_image[loop_i][j] = gaussian_image[i][loop_j]
                    mat_of_g_img[i][loop_j] = gaussian_image[i][loop_j]

                    between_med_seam_region_grayscale[loop_i][j] = med_seam_image[loop_i][j]

                    if travers_pointer.line_binary[loop_i][j] > 20_000:
                        between_med_seam_region_grayscale[loop_i][j] = 255

                    med_seam_region[loop_i][j] = g_img_data[i * step + loop_j * channels]
                    between_med_seam_region_binary[loop_i][j] = med_seam_region[loop_i][j]

                    j += 1
                    loop_j += 1
                loop_i += 1

            line_number_char = line_number

            for i in range(end_col - start_col + 1):
                for j in range(end_row - start_row + 1):
                    bin_line_transpose[i][j] = med_seam_region[j][i]

            derivative_image_transpose = new_generative_derivative_image(
                med_seam_gaussian_image,
                end_row - start_row,
                end_col - start_col
            )

            seam_curve_horizontal = advanced_generation_seam_curve_image_with_binary_info(
                derivative_image_transpose,
                end_col - start_col,
                end_row - start_row,
                bin_line_transpose,
                travers_pointer
            )

            file_name_of_x = f'{output_src}Line-segmentation_line{line_number_string}_x.txt'
            file_name_of_y = f'{output_src}Line-segmentation_line{line_number_string}_y.txt'

            fpx = open(file_name_of_x, "a")
            fpy = open(file_name_of_y, "a")

            last_coordinate_x, last_coordinate_y = 0, 0

            for i in range(start_col, end_col + 1):
                seam_j = 0

                for j in range(start_row, end_row):
                    seam_i = 0

                    if seam_curve_horizontal[seam_i][seam_j] == 255:
                        seam_gray_img[i * step + i * channels] = 0
                        final_seam[j][i] = 255

                        fpx.write(f'{i}\n')
                        fpy.write(f'{j}\n')

                        last_coordinate_x = j
                        last_coordinate_y = i

                        only_seam_data[i * step + j * channels] = 0

                    if seam_curve_horizontal[seam_i][seam_j] == 100:
                        seam_curve_dist[j][i] = 10

                    seam_i += 1
                seam_j += 1

            fpx.write(f'{last_coordinate_x}\n')
            fpy.write(f'{last_coordinate_y}\n')

            fpx.close()
            fpy.close()

            file_name_of_x = f'{output_src}Line-segmentation_line0_x.txt'
            file_name_of_y = f'{output_src}Line-segmentation_line0_y.txt'

            fpx = open(file_name_of_x, "a")
            fpy = open(file_name_of_y, "a")

            for i in range(width):
                fpx.write(f'{i}\n')
                fpy.write('1\n')

            fpx.close()
            fpy.close()

            line_number += 1
            line_number_string = line_number

            file_name_of_x = f'{output_src}Line-segmentation_line{line_number_string}_x.txt'
            file_name_of_y = f'{output_src}Line-segmentation_line{line_number_string}_y.txt'

            fpx = open(file_name_of_x, "a")
            fpy = open(file_name_of_y, "a")

            for i in range(width):
                fpx.write(f'{i}\n')
                fpy.write(f'{raw_img_h - 1}\n')

            fpx.close()
            fpy.close()

            # cv2.imwrite(f'{output_src}gaussian_image_mat.bmp', mat_of_gaussian)

            travers_pointer = head

            medial_seam_first = travers_pointer.start_row

            while travers_pointer.next:
                temp = travers_pointer.next
                medial_seam_last = travers_pointer.end_row
                travers_pointer = temp

        # printf('sukalpa')

        final_seam_rows, final_seam_columns, final_seam_channels = final_seam.shape

        for i in range(final_seam_rows):
            for j in range(final_seam_columns):
                if final_seam[i][j] == 255:
                    inner_i = i - 3
                    while inner_i < i:
                        original_color_mat_image[inner_i][j][0] = 0
                        original_color_mat_image[inner_i][j][1] = 150
                        original_color_mat_image[inner_i][j][2] = 0

                        inner_i += 1

        cv2.imwrite(f'{output_src}Image-with-seam/Separating-green_seam', original_color_mat_image)

        only_seam_mat_image = np.zeros((raw_img_h, raw_img_w))

        # Before anything is done, copy gray image
        binary = g_img.copy()
        original_gray_image = gray_img.copy()

        original_gray_image_rows, original_gray_image_columns, original_gray_image_channels = original_gray_image.shape
        mask = np.zeros((original_gray_image_rows, original_gray_image_columns), np.uint8)
        mask[:] = 256

        for i in range(original_gray_image_rows - 1):
            for j in range(original_gray_image_columns):
                if only_seam_mat_image[i][j] == 0:
                    mask[i][j] = 255
                else:
                    mask[i][j] = 0

        label_image = np.zeros((original_gray_image_rows, original_gray_image_columns), np.uint8)
        n_lables, lable_image, stat_image, centroid = cv2.connectedComponentsWithStats(mask, 8)

        sum_of_width_mask = 0
        for i in range(1, n_lables):
            sum_of_width_mask += stat_image[i][cv2.CC_STAT_WIDTH]

        avg_width_mask = sum_of_width_mask / n_lables

        for i in range(1, n_lables):
            comp_left = stat_image[i][cv2.CC_STAT_LEFT]
            comp_top = stat_image[i][cv2.CC_STAT_TOP]
            comp_width = stat_image[i][cv2.CC_STAT_WIDTH]
            comp_height = stat_image[i][cv2.CC_STAT_HEIGHT]

            if comp_width <= (avg_width_mask / 2):
                for inner_loop_i in range(comp_top, comp_top + comp_height):
                    for inner_loop_j in range(comp_left, comp_left + comp_width):
                        mask[inner_loop_i][inner_loop_j] = 0

        im_thin = np.zeros((g_img_row, g_img_column, 1), np.uint8)

        smooth_final_seam = np.zeros((final_seam_rows, final_seam_columns), np.uint8)

        erosion_size = 1

        element = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1 * erosion_size + 1, 1 * erosion_size + 1),
            (erosion_size, erosion_size)
        )

        eroded_final_seam = cv2.dilate(final_seam, element)

        gray_img = generate_line_image()  # TODO

        dst = np.zeros((raw_img_w, raw_img_h, raw_img_channels), np.uint8)

        beta = 1.0 - 0.5

        fp_xml.write('</SinglePage>\n')

    fp_xml.write('</DocumentList>')
    fp_xml.close()


if __name__ == '__main__':
    image_file = 'test'
    main(image_file)
