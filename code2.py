import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray

from sklearn import svm
from typing import Tuple, Union, List

from gradient_new import gradient
from chain_code_new import chain_code
from tqdm import tqdm
from numba import njit
from PIL import Image

from collections import Counter

import os


columns = 150
num_features = 464
num_samples = 347

# num_samples = 10


def it_over_dir(dir: str):
    X = np.empty((1, num_features), dtype=np.float64)
    y = np.empty(1, dtype=np.uint16)

    correct_class = list()

    error_files: List[Tuple[Union[Image, str], Exception]] = list()

    for i in tqdm(range(1, num_samples + 1), ncols=columns, desc=dir):
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

                correct_class.append(i)

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

    return X, y, correct_class


def main():
    svms = [
        ('SVC', svm.SVC(gamma=18.0)),
        ('SVC_k_linear', svm.SVC(kernel='linear', gamma=18.0)),
        ('SVC_k_rbf', svm.SVC(kernel='rbf', gamma=18.0)),
        # ('SVC_k_poly', svm.SVC(kernel='poly', gamma=18.0)),
        # ('SVC_k_sigmoid', svm.SVC(kernel='sigmoid', gamma=18.0)),
        ('linear_SVC', svm.LinearSVC(C=18.0)),
        ('Nu_SVC', svm.NuSVC(gamma=18.0)),
        # ('linear_SVC_C', svm.LinearSVC(C=18.0)),
        # ('linear_SVC_C_hinge', svm.LinearSVC(C=18.0, loss='hinge')),
        # ('linear_SVC_penalty_l1', svm.LinearSVC(penalty='l1')),
        # ('linear_SVC_penalty_C', svm.LinearSVC(penalty='l1', C=18.0))
        # ('SVR', svm.SVR()),
        # ('linear_SVR', svm.LinearSVR()),
        # ('Nu_SVR', svm.NuSVR()),
    ]

    X, y, correct_classes = it_over_dir("train")

    for kernel, s in tqdm(svms, ncols=columns, desc='Fitting'):
        s.fit(X, y)

    test_X, test_y, correct_class = it_over_dir("test")

    scores = []

    for kernel, s in tqdm(svms, ncols=columns, desc='Calc scores'):
        acc, _ = accuracy(test_X, test_y, correct_class, s)

        scores.append((kernel, acc))
        """
            score = s.score(test_X, test_y)
            scores.append((kernel, score))
        """

        print('Scores:')
        for kernel, score in scores:
            print(f'- {kernel}: {score}')


def accuracy(X: ndarray, y: ndarray, correct_class: List, s: svm.SVC):
    # class_prediction = np.zeros((num_samples, 1))
    class_predict = {}
    prediction = 0

    for i in correct_class:
        if i not in class_predict:
            class_predict[i] = []

    # Append all predictions to list, each element is one image.
    for i, word in enumerate(X):
        w = word.reshape(1, -1)

        predict = s.predict(w)

        if correct_class[i] in class_predict:
            class_predict[correct_class[i]].append(predict[0])

    for i, (key, val) in enumerate(class_predict.items()):
        c = Counter(val)
        value, count = c.most_common()[0]

        class_predict[key] = value

        prediction += 1 if value == key else 0

    return prediction / num_samples, class_predict


if __name__ == '__main__':
    main()
