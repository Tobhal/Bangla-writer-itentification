import numpy as np
from numba import njit


@njit
def least(arr):
    m = arr[0]
    for i in range(300):
        if arr[i] < m:
            m = arr[i]
    return m


@njit
def greatest(arr):
    m = 0
    for i in range(300):
        if arr[i] > m:
            m = arr[i]
    return m


@njit
def division(k, a, dim):
    j, prev = 0, 0
    p = k / dim
    for i in range(dim):
        j = j + p
        q = j
        t = q + 0.5
        if j > t:
            a[i] = q + 1 - prev
        else:
            a[i] = q - prev
        prev = prev + a[i]
    return a


@njit
def h_from_image(image):
    img, (row, col) = image, image.shape

    u = col - 1
    y1 = row - 1

    c1, c2, c3, c4 = np.zeros(2300, ), np.zeros(2300, ), np.zeros(2300, ), np.zeros(2300, )

    l1, l2, l3, l4 = 0, 0, 0, 0

    c1.fill(999)
    c2.fill(0)
    c3.fill(999)
    c4.fill(0)

    for i in range(row):
        for j in range(col):
            if img[i][j] == 1:
                c1[l1] = j
                l1 += 1
                break

        for j in range(u, -1, -1):
            if img[i][j] == 1:
                c2[l2] = j
                l2 += 1
                break

    for i in range(col):
        for j in range(row):
            if img[j][i] == 1:
                c3[l3] = j
                l3 += 1
                break

        for j in range(y1, -1, -1):
            if img[j][i] == 1:
                c4[l4] = j
                l4 += 1
                break

    return int(least(c1)), int(greatest(c2)), int(least(c3)), int(greatest(c4))
