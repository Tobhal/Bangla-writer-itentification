import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import svm
from typing import Tuple, Union, List

from gradient_new import gradient
from chain_code_new import chain_code
from tqdm import tqdm
from numba import njit
from PIL import Image

import os


def main():
    num_features = 464
    num_samples = 347

    X = np.zeros((num_samples, num_features))

    y = np.arange(1, num_samples + 1)

    gradients = np.zeros((num_samples, 400))

    chain_codes = np.zeros((num_samples, 64))

    error_files: List[Tuple[Union[Image, str], Exception]] = list()

    i = 338

    # img_name = f'Bangla-writer-test/test_{i:03}.tif'
    img_name = f'train/train_001/wordseg_line0_word0.png'

    img = Image.open(img_name)

    # convert to binary image
    img = img.convert('L')
    img = img.point(lambda x: 255 if x < 127 else 0, 'L')

    img_gradient = np.asarray(img)
    img_chain_code = np.asarray(img)

    img_gradient = gradient(img_gradient)
    img_chain_code = chain_code(img_chain_code)
    features_combined = np.concatenate((img_gradient, img_chain_code))

    X[i - 1] = features_combined
    gradients[i - 1] = img_gradient
    chain_codes[i - 1] = img_chain_code
    """

    for i in tqdm(range(1, 5 + 1), ncols=100):
        img_name_bw = f'Bangla-writer-test-bw/test_{i:03}.tif'
        img_name = f'Bangla-writer-test/test_{i:03}.tif'

        try:
            if not os.path.isfile(img_name_bw):
                img = Image.open(img_name)

                # convert to binary image
                img = img.convert('L')
                img = img.point(lambda x: 255 if x < 127 else 0, 'L')

                img.save(f'Bangla-writer-test-bw/test_{i:03}.tif')
            else:
                img = Image.open(img_name_bw)

            img_gradient = np.asarray(img)
            img_chain_code = np.asarray(img)

            img_gradient = gradient(img_gradient)
            img_chain_code = chain_code(img_chain_code)
            features_combined = np.concatenate((img_gradient, img_chain_code))

            X[i-1] = features_combined
            gradients[i-1] = img_gradient
            chain_codes[i-1] = img_chain_code
        except Exception as e:
            error_files.append((img_name, e))

    """
    for img, error in error_files:
        print(f'Error: file "{img}" with error: {error}')

    clf = svm.SVC()
    clf.fit(X, y)

    print(f'Score: {clf.score(X, y)}')


if __name__ == '__main__':
    main()
