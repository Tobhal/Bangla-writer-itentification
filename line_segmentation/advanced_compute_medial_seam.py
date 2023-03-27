from typing import List, Tuple, Any

import cv2
import numpy as np
from numpy import ndarray

from types import Image


def gaussian_smooth(
        one_d_pointS_input: Image
) -> Image:
    """
    c++ file: Advanced Compute-Medial Seam- Sukalpa-Nick-Fusion Approach.cpp
    func: void gaussianSmooth
    """
    one_d_pointS_input_rows, one_d_pointS_input_cloumns = one_d_pointS_input.shape

    curve: List[Tuple[int, Any]] = [(i, one_d_pointS_input[i][0]) for i in range(one_d_pointS_input_rows)]

    sigma = 13.0
    M = ((10.0 * sigma + 1.0) / 2.0) * 2
    assert M % 2 == 1

    d, dg, d2g = get_gaussian_derivs(sigma, M)

    curve_x, curve_y = poly_line_split(curve)

    X, XX =



def get_gaussian_derivs(
        sigma: float,
        M: int
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    c++ file: Advanced Compute-Medial Seam- Sukalpa-Nick-Fusion Approach.cpp
    func: void getGaussianDerivs
    """
    L = int((M - 1) / 2)
    sigma_squared = sigma**2
    sigma_quad = sigma**4

    gaussian = np.zeros((M, 1), np.double)
    dg = np.zeros((M, 1), np.double)
    d2g = np.zeros((M, 1), np.double)

    g = cv2.getGaussianKernel(M, sigma, np.float64)

    i = -L
    while i < L + 1.0:
        idx = int(i + L)
        gaussian[idx] = g(idx)
        dg[idx] = (-i / sigma_squared) * g(idx)
        d2g = (-sigma_squared + i * i) / sigma_quad * g(idx)

        i += 1.0

    return gaussian, dg, d2g


def poly_line_split(
    pl: List[Tuple[int, Any]]
) -> Tuple[ndarray, ndarray]:
    """
    c++ file: Advanced Compute-Medial Seam- Sukalpa-Nick-Fusion Approach.cpp
    func: void PolyLineSplit
    """
    contour_x = np.zeros((len(pl), 1), np.float)
    contour_y = np.zeros((len(pl), 1), np.float)

    for i in range(len(pl)):
        contour_x[i] = float(pl[i][0])
        contour_y[i] = float(pl[i][1])

    return contour_x, contour_y


def get_X_curve(
        x: List[float],
        sigma: float,
        g: ndarray,
        dg: ndarray,
        d2g: ndarray,
        is_open: bool
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    c++ file: Advanced Compute-Medial Seam- Sukalpa-Nick-Fusion Approach.cpp
    func: void getXcurve
    """
    gx = np.zeros((len(x), 1), np.double)
    dx = np.zeros((len(x), 1), np.double)
    d2x = np.zeros((len(x), 1), np.double)

    for i in range(len(x)):
        gaus_x = 0.0
        dg_x = 0.0
        d2g_x = 0.0



def get_d_x(

):
    """
    c++ file: Advanced Compute-Medial Seam- Sukalpa-Nick-Fusion Approach.cpp
    func: void getdX
    """