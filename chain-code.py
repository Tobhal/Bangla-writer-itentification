import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import unittest

from numpy import ndarray

from util import h_from_image, division


def reduction_from_49_to_16(b, count):
    nc1 = np.zeros((9, 9, 4))
    nc2 = np.zeros((5, 5, 16))
    app = np.zeros((64,))
    nc = b
    h = 0

    for i in range(0, 7, 2):
        for j in range(0, 7, 2):
            for k in range(4):
                nc[i][j][k] = nc[i][j][k] * 2

    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(4):
                nc1[i][j][k] = nc[i - 1][j - 1][k]

    for i in range(1, 8, 2):
        for j in range(1, 8, 2):
            for k in range(4):
                nc2[int(i / 2)][int(j / 2)][k] = nc1[i][j][k] + nc1[i - 1][j - 1][k] + nc1[i - 1][j][k] + \
                                                 nc1[i - 1][j + 1][k] + nc1[i][j - 1][k] + nc1[i][j + 1][k] + \
                                                 nc1[i + 1][j - 1][k] + nc1[i + 1][j][k] + nc1[i + 1][j + 1][k]

    for i in range(4):
        for j in range(4):
            for k in range(4):
                nc2[i][j][k] = nc2[i][j][k] / count

    for i in range(4):
        for j in range(4):
            for k in range(4):
                app[h] = nc2[i][j][k]
                h = h + 1

    return app


def chain_code(image) -> ndarray:
    """
    Main function
    """
    img = Image.open(image)
    print(img.size)
    img = np.asarray(img)
    img = 255.0 - img
    print(img.shape)
    print(img[0])
    img = img / 255.0
    plt.imshow(img, cmap='Greys')
    plt.show()

    image_a = img
    b1 = image_a

    b = np.zeros((7, 7, 4))

    h1, h2, h3, h4 = h_from_image(img)

    ro = h4 - h3 + 1
    co = h2 - h1 + 1

    r = np.zeros((7,))
    c = np.zeros((7,))

    r = division(ro, r, 7)
    c = division(co, c, 7)

    h8 = h3 + 1
    h9 = h1 + 1
    for i in range(h8, h4, 1):
        for j in range(h9, h2, 1):
            if image_a[i][j] == 0:
                b1[i][j] = 0
            else:
                if image_a[i][j] == 1:

                    if image_a[i][j + 1] == 1 and image_a[i - 1][j + 1] == 1 and image_a[i - 1][j] == 1 and \
                            image_a[i - 1][
                                j - 1] == 1 and image_a[i][j - 1] == 1 and image_a[i + 1][j - 1] == 1 and \
                            image_a[i + 1][j] == 1 and \
                            image_a[i + 1][j + 1] == 1:
                        b1[i][j] = 1
                    else:
                        b1[i][j] = 3

    for i in range(h3, h4 + 1):
        if image_a[i][h1] == 0:
            b1[i][h1] = 0
        else:
            b1[i][h1] = 3

        if image_a[i][h2] == 0:
            b1[i][h2] = 0
        else:
            b1[i][h2] = 3

    for j in range(h1, h2 + 1):
        if image_a[h3][j] == 0:
            b1[h3][j] = 0
        else:
            b1[h3][j] = 3

        if image_a[h4][j] == 0:
            b1[h4][j] = 0
        else:
            b1[h4][j] = 3

    for m in range(0, 7):
        for n in range(0, 7):
            s = h1
            y = h3

            for i in range(m):
                y = y + r[i]
            z = y + r[m]

            for j in range(0, n):
                s = s + c[j]
            t = s + c[n]

            z, y, t, s = int(z), int(y), int(t), int(s)

            for i in range(y, z):
                for j in range(s, t):
                    if b1[i][j] == 0 or b1[i][j] == 1:
                        continue
                    else:
                        if (j + 1) < t:
                            if b1[i][j + 1] == 3:
                                b[m][n][0] = b[m][n][0] + 1
                        if (i - 1) >= y and (j + 1) < t:
                            if b1[i - 1][j + 1] == 3:
                                b[m][n][1] = b[m][n][1] + 1
                        if (i - 1) >= y:
                            if b1[i - 1][j] == 3:
                                b[m][n][2] = b[m][n][2] + 1
                        if (i - 1) >= y and (j - 1) >= s:
                            if b1[i - 1][j - 1] == 3:
                                b[m][n][3] = b[m][n][3] + 1

    count = 0
    for i in range(h3, h4 + 1):
        for j in range(h1, h2 + 1):
            if b1[i][j] == 3:
                count += 1

    app = reduction_from_49_to_16(b, count)

    b = np.asarray(b)
    print(b.shape)

    app = np.asarray(app)
    print(app.shape)
    print(app)

    return app

"""Test's
class MyTestCase(unittest.TestCase):
    def test_something(self):
        correct_value = [
            0.11211571596989209, 0.10684683050693752, 0.11204316677246758, 0.10627550557721956, 0.15130135122880203,
            0.14381064659472204, 0.15107463498685045, 0.14238686859526617, 0.12480275686950214, 0.11747528792962728,
            0.1230978507300263, 0.11567062664369275, 0.09161149904779178, 0.08729482180103383, 0.09224630452525619,
            0.08610682869320758, 0.1543937607690215, 0.14672168314138026, 0.15436655481998732, 0.1475650675614401,
            0.20671080076176657, 0.1980955835676068, 0.2080801668631541, 0.19533871406547565, 0.1691212478461957,
            0.1610682869320758, 0.17021855445724132, 0.1581572503854176, 0.14024666727124332, 0.13255645234424593,
            0.1401469121247846, 0.1301713974789154, 0.13825156434206948, 0.1313412532873855, 0.14040990296544845,
            0.1334995919107645, 0.18403010791693117, 0.17529699827695655, 0.18461050149632718, 0.17409993651945224,
            0.1812823070644781, 0.17214110818899067, 0.18126416976512197, 0.16941144463589372, 0.1323206674526163,
            0.1261358483721774, 0.1338442005985309, 0.12352407726489525, 0.09777818082887459, 0.09308968894531604,
            0.10021764759227351, 0.09622744173392582, 0.12508388500952208, 0.11788337716514011, 0.12500226716241952,
            0.11781082796771561, 0.11256914845379523, 0.10681055590822526, 0.11196154892536501, 0.10358211662283485,
            0.07261267797224993, 0.06896708080166863, 0.07186904869864877, 0.06626462319760587
        ]

        app = chain_code('Bangla-writer-test/test_001.tif')

        for _i in range(len(app)):
            self.assertEqual(app[_i], correct_value[_i])
"""

if __name__ == '__main__':
    #unittest.main()
    pass
