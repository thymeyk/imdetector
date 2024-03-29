import inspect

import joblib
import cv2 as cv
import numpy as np
from enum import IntEnum
from multiprocessing import Pool

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC, OneClassSVM


class Color(IntEnum):
    B = 0
    G = 1
    R = 2
    Y = 3
    Cr = 4
    Cb = 5


class DrawFlags(IntEnum):
    SHOW_RESULT = 0
    SHOW_FULL_RESULT = 1
    RETURN_RESULT = 2


class BaseDetector:
    """
    Base class for all detectors
    """

    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        return sorted([p.name for p in parameters])

    def get_params(self):
        """
        Get parameters for this estimator.
        """

        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None
            # if deep and hasattr(value, 'get_params'):
            #     deep_items = value.get_params().items()
            #     out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def __repr__(self):
        params = self.get_params()
        repr_ = ''
        for name, val in params.items():
            repr_ += '{}={}, '.format(name, val)
        repr_ = repr_.rsplit(',', 1)[0]
        return '{}({})'.format(self.__class__.__name__, repr_)

    def save_image(self, file_name):
        if hasattr(self, 'image_'):
            cv.imwrite(file_name, self.image_)
        else:
            print('Error: There is no result image.')
        return self


class BaseDetectorMachine(BaseDetector):
    """
    Base class for all detectors using machine learning
    """

    def __init__(self,
                 feature_extractor,
                 model_name,
                 trainable=False,
                 flags=DrawFlags.SHOW_RESULT,
                 **kwargs):
        self.feature_extractor = feature_extractor(**kwargs)
        self.model_name = model_name
        self.flags = flags
        if not trainable:
            self.clf = joblib.load(self.model_name)

    def detect(self, imgs):
        """
        :param imgs: list of SuspiciousImage
        :return: Suspect(1) or not(0)
        :rtype: int
        """

        if isinstance(imgs, list):
            # X = np.stack([self.feature_extractor.extract(i) for i in imgs])
            X = self.feature_extractor.extract(imgs)
            return X
        else:
            print("ERROR: unsupported input type")

    def fit_img(self, train_img, train_y, test_img, test_y,
                model, C, gamma, nu, n_estimators, max_depth):
        train_X = self.feature_extractor.extract(train_img)
        test_X = self.feature_extractor.extract(test_img)
        self.fit_X(train_X, train_y, test_X, test_y,
                   model, C, gamma, nu, n_estimators, max_depth)

    def fit_X(self, train_X, train_y, test_X, test_y,
              model, C, gamma, nu, n_estimators, max_depth):
        if model == 'svm':
            probability = (self.flags == DrawFlags.SHOW_RESULT
                           or self.flags == DrawFlags.SHOW_FULL_RESULT)
            self.clf = SVC(C=C,
                           gamma=gamma,
                           probability=probability)
        elif model == 'onesvm':
            self.clf = OneClassSVM(gamma=gamma, nu=nu)
        elif model == 'rf':
            self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth)
        self.clf.fit(train_X, train_y)
        pred = self.clf.predict(test_X)
        print(
            classification_report(
                test_y,
                pred,
                target_names=[
                    'Safe',
                    'Suspect']))

    def save_images(self, file_names):
        if hasattr(self, 'image_') and self.image_ is not None:
            [cv.imwrite(file_name, image_)
             for file_name, image_ in zip(file_names, self.image_)]
        else:
            print(self, 'Error: There is no result image.')
        return self


class BaseFeatureExtractor:
    def extract(self, imgs):
        """
        :param list imgs: suspicious image(s)
        :return: X, feature vector of suspect image
        :rtype: np.ndarray
        """

        # X = np.stack([self.feature(i.mat[4:-4, 4:-4, :]) for i in imgs])
        # X = np.stack([self.feature(i.mat) for i in imgs])
        p = Pool(self.n_jobs)
        X = p.map(self.feature, [i.mat for i in imgs])
        p.close()
        X = np.stack(X)

        return X

    def feature(self, img):
        """
        :param np.ndarray img: suspicious image
        :return: X, feature matrix of suspect image(s)
        :rtype: np.ndarray
        """
        return img

    def save_features(self, img, file_name):
        X = self.extract(img)
        if not isinstance(X, int):
            np.savetxt(file_name, X, delimiter=',')
        return self
