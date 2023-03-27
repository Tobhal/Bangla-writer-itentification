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


columns = 150


def it_over_dir(dir: str, num_features: int, num_samples: int):
    X = np.empty((1, num_features), dtype=np.float64)
    y = np.empty(1, dtype=np.uint16)

    error_files: List[Tuple[Union[Image, str], Exception]] = list()

    for i in tqdm(range(1, num_samples + 1), ncols=columns, desc=dir):
    # for i in range(1, num_samples + 1):
        folder = f'{dir}/{dir}_{i:03}'

        if not os.path.isdir(folder):
            continue

        files = os.listdir(folder)

        wordseg = [f for f in files if f.startswith('wordseg_')]
        wordseg.sort()

        for word in wordseg:
            try:
                img = Image.open(f'{folder}/{word}')

                # convert to binary image
                img = img.convert('L')
                img = img.point(lambda x: 255 if x < 127 else 0, 'L')

                img = np.asarray(img)

                img_gradient = np.asarray(img)
                img_chain_code = np.asarray(img)

                img_gradient = gradient(img_gradient)
                img_chain_code = chain_code(img_chain_code)
                features_combined = np.concatenate((img_gradient, img_chain_code))

                features_combined = features_combined.reshape(1, -1)

                X = np.append(X, features_combined, axis=0)
                y = np.append(y, i)

            # except Exception as e:
            except FileNotFoundError as e:
                error_files.append((word, e))

    for img, error in error_files:
        print(f'Error: file "{img}" with error: {error}')

    X = np.asarray(X)
    y = np.asarray(y)

    X = np.delete(X, 0, axis=0)
    y = np.delete(y, 0)

    # print(y.shape)
    # print(X.shape)

    return X, y


def main():
    num_features = 464
    num_samples = 347

    # SVMs
    svms = [
        # ('linear', svm.SVC(kernel='linear', gamma=18.0)),
        # ('rbf', svm.SVC(kernel='rbf', gamma=18.0)),
        # ('poly', svm.SVC(kernel='poly', gamma=18.0)),
        # ('sigmoid', svm.SVC(kernel='sigmoid', gamma=18.0)),
        # ('linear_SVC_C', svm.LinearSVC(C=18.0)),
        # ('linear_SVC_C_hinge', svm.LinearSVC(C=18.0, loss='hinge')),
        # ('linear_SVC_penalty_l1', svm.LinearSVC(penalty='l1')),
        # ('linear_SVC_penalty_C', svm.LinearSVC(penalty='l1', C=18.0))
        ('SVC', svm.SVC()),
        ('linear_SVC', svm.LinearSVC()),
        # ('Nu_SVC', svm.NuSVC()),
        ('SVR', svm.SVR()),
        ('linear_SVR', svm.LinearSVR()),
        # ('Nu_SVR', svm.NuSVR()),
    ]

    X, y = it_over_dir("train", num_features, num_samples)

    for kernel, s in tqdm(svms, ncols=columns, desc='Fitting'):
        s.fit(X, y)

    test_X, test_y = it_over_dir("test", num_features, num_samples)

    scores = []

    for kernel, s in tqdm(svms, ncols=columns, desc='Calc scores'):
        score = s.score(test_X, test_y)
        scores.append((kernel, score))

    print('Scores:')
    for kernel, score in scores:
        print(f'- {kernel}: {score}')


if __name__ == '__main__':
    main()
