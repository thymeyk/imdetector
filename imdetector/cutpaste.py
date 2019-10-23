import pywt
import cv2 as cv
import numpy as np
from itertools import chain
from multiprocessing import Pool

from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from imdetector.utils import extract_nonoverlap_patches
from .base import BaseDetector, DrawFlags, Color
from .image import SuspiciousImage


class DWTFeatureExtractor:
    """
    Parameters
    ----------
    channel : int, (default=Color.B)
        See base.Color.
    block_size : int, (default=0)
        Length of one side of the block.
    t : int, (default=4)
        Truncation threshold.
    n_jobs : int, (default=1)
    """

    def __init__(self, channel=Color.B, block_size=0, t=4, n_jobs=1):
        self.channel = channel
        self.t = t
        self.block_size = block_size
        self.n_jobs = n_jobs

    def extract(self, img):
        """
        :param list | np.ndarray | SuspiciousImage img: suspicious image(s)
        :return: X, feature matrix of suspicious image(s)
        :rtype: np.ndarray
        """

        if isinstance(img, SuspiciousImage):
            X = np.array(self.feature(img.mat))
        elif isinstance(img, np.ndarray):
            X = np.array(self.feature(img))
        elif isinstance(img, list):
            X = np.array([self.feature(imarray) for imarray in img])
        else:
            print("ERROR: unsupported input type")
            return 0
        return X

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
    def transition_probability(matrix):
        Wk, Dhk, Dvk, t = matrix
        Mhh = [0 if i == 0
               else max(0,
                        np.sum((Dhk[:-1, :] == i) & (np.diff(Dhk, axis=0) == (j - i)))
                        / np.sum(Dhk[:-1, :] == i))
               for i in range(-t, t + 1) for j in range(-t, t + 1)]
        Mhv = [0 if i == 0
               else max(0,
                        np.sum((Dhk[:, :-1] == i) & (np.diff(Dhk, axis=1) == (j - i)))
                        / np.sum(Dhk[:, :-1] == i))
               for i in range(-t, t + 1) for j in range(-t, t + 1)]
        Mvh = [0 if i == 0
               else max(0,
                        np.sum((Dhk[:, :-1] == i) & (np.diff(Dhk, axis=1) == (j - i)))
                        / np.sum(Dhk[:, :-1] == i))
               for i in range(-t, t + 1) for j in range(-t, t + 1)]
        Mvv = [0 if i == 0
               else max(0,
                        np.sum((Dvk[:, :-1] == i) & (np.diff(Dvk, axis=1) == (j - i)))
                        / np.sum(Dvk[:, :-1] == i))
               for i in range(-t, t + 1) for j in range(-t, t + 1)]
        return Mhh + Mhv + Mvh + Mvv

    def feature(self, img):
        """
        :param np.ndarray img: suspicious image
        :return: X, feature matrix of suspect image(s)
        :rtype: np.ndarray
        """

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
        p = Pool(self.n_jobs)
        M = p.map(self.transition_probability,
                  [[W[k], Dh[k], Dv[k], self.t] for k in range(13)])
        p.close()

        X = np.array(chain.from_iterable(M))
        X = X.flatten()

        return X

    def save_features(self, img, file_name):
        X = self.extract(img)
        if not isinstance(X, int):
            np.savetxt(file_name, X, delimiter=',')
        return self


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
        :return: X, feature matrix of suspect image(s)
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


class CutPaste(BaseDetector):
    """
    Parameters
    ----------
    feature_extractor : FeatureExtractor class, (default=DWTFeatureExtractor)
    model_name : str,
        Path to trained model.

    Attributes
    ----------
    clf_ : classifier,
    pred_proba_ : float,
    """

    def __init__(self,
                 feature_extractor=DWTFeatureExtractor,
                 model_name='../model/cutpaste_svm.sav',
                 flags=DrawFlags.SHOW_RESULT):
        self.feature_extractor = feature_extractor()
        self.model_name = model_name
        self.flags = flags

    def detect(self, img):
        """
        :param img:
        :return: Suspect(1) or not(0)
        :rtype: int
        """

        if hasattr(self, 'clf_'):
            self.clf_ = joblib.load(self.model_name)

        X = self.feature_extractor.extract(img)

        pred = self.clf_.predict(X)

        if self.flags == DrawFlags.SHOW_RESULT or self.flags == DrawFlags.SHOW_FULL_RESULT:
            self.pred_proba_ = self.clf_.predict_proba(X)

        return pred

    def load_model(self, model_name):
        self.clf_ = joblib.load(model_name)

    def fit_img(self, train_img, train_y, test_img, test_y):
        train_X = self.feature_extractor.extract(train_img)
        test_X = self.feature_extractor.extract(test_img)
        self.fit_X(train_X, train_y, test_X, test_y)

    def fit_X(self, train_X, train_y, test_X, test_y,
              model='svm', C=1000, gamma=0.001,
              n_estimators=1000, max_depth=50):
        if model == 'svm':
            probability = (self.flags == DrawFlags.SHOW_RESULT
                           or self.flags == DrawFlags.SHOW_FULL_RESULT)
            self.clf_ = SVC(C=C,
                            gamma=gamma,
                            probability=probability)
        elif model == 'rf':
            self.clf_ = RandomForestClassifier(n_estimators=n_estimators,
                                               max_depth=max_depth)
        self.clf_.fit(train_X, train_y)
        pred = self.clf_.predict(test_X)
        print(
            classification_report(
                test_y,
                pred,
                target_names=[
                    'Safe',
                    'Suspect']))
