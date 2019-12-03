import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from imdetector.image import SuspiciousImage
from imdetector.base import BaseDetectorMachine, BaseFeatureExtractor, DrawFlags


class NoiseFeatureExtractor(BaseFeatureExtractor):
    """
    Parameters
    ----------
    size : int, (default=256)
        Length of one side of image.
    t : int, (default=4)
        Truncation threshold.
    """

    def __init__(self, size=256, t=4):
        self.size = size
        self.t = t

    def extract(self, imgs):
        """
        :param list | np.ndarray | SuspiciousImage imgs: suspicious image(s)
        :return: X, feature matrix of suspicious image(s)
        :rtype: np.ndarray
        """

        X = np.stack([self.feature(i.gray) for i in imgs])

        return X

    def feature(self, img):
        """
        :param np.ndarray img: laplacian of suspicious image
        :return: X, feature vector of suspect image
        :rtype: np.ndarray
        """

        lap = cv.resize(img, dsize=(self.size, self.size))
        lap = abs(cv.filter2D(lap, -1,
                              np.array([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], np.float32), delta=100).astype('int') - 100)
        lap = np.where(lap > self.t, self.t, lap) / self.t
        lap = lap.flatten()
        return lap


class Noise(BaseDetectorMachine):
    """
    Parameters
    ----------
    feature_extractor : FeatureExtractor class, (default=NoiseFeatureExtractor)
    model_name : str,
        Path to trained model.
    trainable : bool, (default=False)
    size : int, (default=256)
        Length of one side of image.
    color : Tuple[int, int, int], (default=(0,255,255))

    Attributes
    ----------
    dist_ : array-like, shape (n_samples,)
        Signed distance to the separating hyperplane.
    """

    def __init__(
            self,
            feature_extractor=NoiseFeatureExtractor,
            model_name='./model/noise_oneclass_42.sav',
            trainable=False,
            size=256,
            color=(0, 255, 255),
            flags=DrawFlags.SHOW_RESULT):
        super().__init__(feature_extractor, model_name, trainable, flags)
        self.size = size
        self.color = color

    def detect(self, imgs):
        """
        :param imgs: list of SuspiciousImage
        :return: Suspect(1) or not(0)
        :rtype: int
        """
        self.image_ = []

        X = super().detect(imgs)

        pred = self.clf.predict(X)
        pred = np.where(pred == -1, 0, pred)

        if self.flags != DrawFlags.RETURN_RESULT:
            self.dist_ = self.clf.decision_function(X)

        if self.flags == DrawFlags.SHOW_FULL_RESULT:
            for i, p in enumerate(pred):
                img = imgs[i]
                img_noise = np.where(img.lap > 4, 4, img.lap) / 4 * 255
                img_noise = (255 - img_noise)
                img_noise = cv.cvtColor(
                    img_noise.astype('uint8'), cv.COLOR_GRAY2BGR)
                if p:
                    img_noise = cv.rectangle(
                        img_noise, (0, 0), img.gray.T.shape, self.color, thickness=5)
                self.image_.append(img_noise)
                plt.close()

        return pred

    def fit_X(self, train_X, train_y, test_X, test_y,
              model='onesvm', gamma=0.0001, nu=0.01, **kwargs):
        super().fit_X(train_X, train_y, test_X, test_y,
                      model=model, gamma=gamma, nu=nu, **kwargs)
