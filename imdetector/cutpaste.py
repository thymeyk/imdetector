import os

import pywt
import cv2 as cv
import numpy as np
from itertools import chain
from multiprocessing import Pool

from skimage.feature import local_binary_pattern

from utils import extract_nonoverlap_patches
from base import BaseFeatureExtractor, DrawFlags, Color, BaseDetectorMachine


class DWTFeatureExtractor(BaseFeatureExtractor):
    """
    Parameters
    ----------
    channel : int, (default=Color.B)
        See base.Color.
    block_size : int, (default=0)
        Length of one side of block.
    t : int, (default=4)
        Truncation threshold.
    n_jobs : int, (default=1)
    """

    def __init__(self, channel=Color.B, block_size=0, t=4, n_jobs=4):
        self.channel = channel
        self.t = t
        self.block_size = block_size
        self.n_jobs = n_jobs

    @staticmethod
    def block_dwt(b, wavelet='dmey'):
        A1, _ = pywt.wavedec2(b, wavelet, level=1)
        A2, _, _ = pywt.wavedec2(b, wavelet, level=2)
        A3, coeffs3, coeffs2, coeffs1 = pywt.wavedec2(b, wavelet, level=3)
        W = [b] + [A1] + list(coeffs1) + [A2] + \
            list(coeffs2) + [A3] + list(coeffs3)
        W = np.array(list(map(lambda x: np.abs(np.round(x)).astype(int), W)))
        return W

    @staticmethod
    def cond_probability(A, B, i, j):
        """
        :param np.ndarray A:
        :param np.ndarray B:
        :param int i:
        :param int j:
        :return:
        """
        denominator = np.sum(A == i)
        if denominator == 0:
            return 0
        numerator = np.sum((A == i) & (B == j))
        return numerator / denominator

    def transition_probability(self, matrix):
        Wk, Dhk, Dvk = matrix
        t = self.t
        Mhh = [self.cond_probability(Dhk[:-1, :], np.diff(Dhk, axis=0), i, j - i)
               for i in range(-t, t + 1) for j in range(-t, t + 1)]
        Mhv = [self.cond_probability(Dhk[:, :-1], np.diff(Dhk, axis=1), i, j - i)
               for i in range(-t, t + 1) for j in range(-t, t + 1)]
        Mvh = [self.cond_probability(Dvk[:-1, :], np.diff(Dvk, axis=0), i, j - i)
               for i in range(-t, t + 1) for j in range(-t, t + 1)]
        Mvv = [self.cond_probability(Dvk[:, :-1], np.diff(Dvk, axis=1), i, j - i)
               for i in range(-t, t + 1) for j in range(-t, t + 1)]
        return Mhh + Mhv + Mvh + Mvv

    def feature(self, img):
        """
        :param np.ndarray img: suspicious image
        :return: X, feature matrix of suspect image(s)
        :rtype: np.ndarray
        """

        # if img.shape[0] < 8 or img.shape[1] < 8:
        #     print("Too small image")
        #     return [0] * (52 * (self.t * 2 + 1)**2)
        # print(img.shape)

        # 1. Take one color channel
        if 3 <= self.channel < 6:
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        r = img[:, :, self.channel]

        # 2. N x N non-overlap blocks & 3-level DWT
        if self.block_size > 0:
            h, w = r.shape
            n = self.block_size

            p = Pool(self.n_jobs)
            block = p.map(self.block_dwt,
                          [r[i * n:(i + 1) * n, j * n:(j + 1) * n]
                           for i in range(h // n) for j in range(w // n)])
            p.close()

            block = np.array(block).T
            W = [np.block(list(map(lambda x: list(x),
                                   b.reshape((h // n, w // n)))))
                 for b in block]
        else:
            W = self.block_dwt(r)

        # 3. Calculate the horizontal and vertical difference arrays
        Dh = [np.clip(-np.diff(wk, axis=0), -self.t, self.t) for wk in W]
        Dv = [np.clip(-np.diff(wk, axis=1), -self.t, self.t) for wk in W]

        # 4. Calculate the horizontal and vertical transition probability
        # matrices
        M = [self.transition_probability(
            [W[k], Dh[k], Dv[k]]) for k in range(13)]

        X = list(chain.from_iterable(M))

        return X


class LBPDCTFeatureExtractor(DWTFeatureExtractor):
    """
    Parameters
    ----------
    channel : int, (default=Color.B)
        See base.Color.
    block_size : int, (default=32)
        Length of one side of the block.
    """

    def __init__(self, channel=Color.B, block_size=32):
        super().__init__(channel=channel, block_size=block_size)

    def feature(self, img):
        """
        :param np.ndarray img: suspicious image
        :return: X, feature vector of suspect image
        :rtype: np.ndarray
        """

        # 1. Take one color channel
        if 3 <= self.channel < 6:
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        r = img[:, :, self.channel]

        # 2. Extract block_size x block_size non-overlap blocks
        blocks = extract_nonoverlap_patches(
            r, patch_size=(self.block_size, self.block_size))

        # 3. Take local binary pattern
        blocks = list(map(lambda x: local_binary_pattern(
            x, 8, 1).astype('float32'), blocks))

        # 4. Apply DCT
        blocks = list(map(lambda x: cv.dct(x), blocks))

        # 5. Calculate STD
        std = np.std(np.array(blocks), axis=0)
        std = (std - np.min(std)) / (np.max(std) - np.min(std)).flatten()

        return std


class CutPaste(BaseDetectorMachine):
    """
    Parameters
    ----------
    feature_extractor : FeatureExtractor class, (default=DWTFeatureExtractor)
    model_name : str,
        Path to trained model.
    trainable : bool, (default=False)

    Attributes
    ----------
    proba_ : array-like, shape (n_samples,)
    """

    def __init__(
            self,
            feature_extractor=DWTFeatureExtractor,
            model_name='./model/cutpaste_svm_yrc_200.sav',
            param_name='./model/cutpaste_svm_yrc_200.sav-param.npz',
            trainable=False,
            flags=DrawFlags.SHOW_RESULT):
        super().__init__(feature_extractor, model_name, trainable, flags)
        self.param_name = param_name

    def detect(self, img):
        """
        :param img:
        :return: Suspect(1) or not(0)
        :rtype: int
        """

        X = super().detect(img)
        if os.path.exists(self.param_name):
            dictionary = np.load(self.param_name)
            support = dictionary['support']
            X = X[:, support]

        pred = self.clf.predict(X)

        if self.flags == DrawFlags.SHOW_RESULT or self.flags == DrawFlags.SHOW_FULL_RESULT:
            self.proba_ = self.clf.predict_proba(X)[:, 1]

        return pred

    def fit_X(self, train_X, train_y, test_X, test_y,
              model='svm', C=1000, gamma=0.001, **kwargs):
        self.param_name = ''
        super().fit_X(train_X, train_y, test_X, test_y,
                      model=model, C=C, gamma=gamma, **kwargs)
