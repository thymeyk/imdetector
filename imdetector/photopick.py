import cv2 as cv
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

from imdetector.image import SuspiciousImage
from imdetector.base import BaseDetectorMachine, BaseFeatureExtractor, DrawFlags


class DictionaryFeatureExtractor(BaseFeatureExtractor):
    """
    See also https://github.com/sephonlee/viziometrics.

    Parameters
    ----------
    size : int, (default=256)
        Length of one side of image.
    t : int, (default=4)
        Truncation threshold.
    """

    def __init__(self, param_name, size=128, block_size=6):
        self.size = size
        self.block_size = block_size
        self.param_name = param_name
        dictionary = np.load(param_name)
        self.M = dictionary['Mean']
        self.P = dictionary['Patch']
        self.centroids = dictionary['centroids']

    def extract(self, imgs):
        """
        :param list imgs: suspicious image(s)
        :return: X, feature vector of suspect image
        :rtype: np.ndarray
        """

        dsize = self.size
        img_arrs = [cv.resize(i.gray, dsize=(dsize, dsize)) for i in imgs]
        X = self.feature(img_arrs)

        return X

    @staticmethod
    def normalize(A):
        pM = np.mean(np.asmatrix(A), axis=1)
        pSqVar = np.sqrt(A.var(axis=1) + 10)
        pSqVar = pSqVar.astype('float64')
        A = np.divide((A - pM), pSqVar)
        return A

    @staticmethod
    def whitening(A):
        C = np.cov(A, rowvar=False)
        M = np.mean(A, axis=0)
        D, V = np.linalg.eig(C)
        P = np.dot(np.dot(V, np.diag(np.sqrt(1 / (D + 0.1)))), V.T)
        A = np.dot(A - M, P)
        return A, M, P

    @staticmethod
    def patch_hist(subX):
        return np.squeeze(np.sum(np.sum(subX, axis=0), axis=0))

    def feature(self, imgs):
        """
        :param np.ndarray imgs: array of suspicious image (BGR)
        :return: X, array of feature vector of suspect image
        :rtype: np.ndarray
        """

        n_imgs = len(imgs)
        n_centroids = self.centroids.shape[0]
        p_size = self.block_size
        # 重心ごとの要素(6*6個)の二乗和
        cc = np.asmatrix(np.sum(np.power(self.centroids, 2), axis=1).T)
        XC = np.zeros((n_imgs, 4 * n_centroids))

        for i in range(n_imgs):
            # extract patches from a image
            ps = extract_patches_2d(imgs[i], (p_size, p_size))
            ps = np.asmatrix(ps.reshape(-1, p_size * p_size))

            # normalize
            ps = np.divide(ps - np.mean(ps, axis=1),
                           np.sqrt(np.var(ps, axis=1) + 1))
            # whitening
            ps = np.dot((ps - self.M), self.P)

            # calcurate the distance z between points x and centroids
            xx = np.sum(np.power(ps, 2), axis=1)
            xc = np.dot(ps, self.centroids.T)
            z = np.sqrt(cc + xx - 2 * xc)  # euclidean distance

            # find the nearest centroids
            # v = np.min(z, axis=1)  # 各パッチから一番近い重心との距離
            inds = np.argmin(z, axis=1)  # 一番近い重心のindex
            mu = np.mean(z, axis=1)  # パッチから各重心の距離の平均
            ps = mu - z
            ps[ps < 0] = 0

            # set a flag 1 to the index of the nearest centroids
            off = np.asmatrix(
                range(
                    0,
                    (z.shape[0]) *
                    n_centroids,
                    n_centroids))
            ps = np.zeros((ps.shape[0] * ps.shape[1], 1))
            ps[off.T + inds] = 1
            ps = np.reshape(ps, (z.shape[1], z.shape[0]), 'F').T

            # split a image into 4 quadrants
            finalDim = [self.size - p_size + 1, self.size - p_size + 1, 1]
            ps = np.reshape(ps, (finalDim[0], finalDim[1], n_centroids), 'F')
            n = np.min(ps.shape[0:2])
            split = int(round(float(n) / 2))

            Q = np.vstack((self.patch_hist(ps[:split, :split, :]),
                           self.patch_hist(ps[split:, :split, :]),
                           self.patch_hist(ps[:split, split:, :]),
                           self.patch_hist(ps[split:, split:, :]))).T

            XC[i, :] = Q.flatten()

        FSMean = np.mean(XC, axis=0)
        FSSd = np.sqrt(np.var(XC, axis=0) + 0.01)

        XCs = np.divide(XC - FSMean, FSSd)

        return XCs


class PhotoPick(BaseDetectorMachine):
    """
    Parameters
    ----------
    feature_extractor : FeatureExtractor class, (default=NoiseFeatureExtractor)
    model_name : str,
        Path to trained model.

    Attributes
    ----------
    clf_ : classifier,
    dist_ : array-like, shape (n_samples,)
        Signed distance to the separating hyperplane.
    """

    def __init__(
            self,
            feature_extractor=DictionaryFeatureExtractor,
            model_name='./model/photopicker_rf_lee_2700.sav',
            param_name='./model/photopicker_rf_lee_2700.sav-param.npz',
            flags=DrawFlags.SHOW_RESULT):
        super().__init__(feature_extractor, model_name, flags, param_name=param_name)

    def detect(self, imgs):
        """
        :param imgs:
        :return: Suspect(1) or not(0)
        :rtype: int
        """

        X = super().detect(imgs)

        pred = self.clf_.predict(X)
        pred = np.where(pred == -1, 0, pred)

        return pred

    def fit_X(self, train_X, train_y, test_X, test_y,
              model='onesvm', gamma=0.0001, nu=0.01, **kwargs):
        super().fit_X(train_X, train_y, test_X, test_y,
                      model=model, gamma=gamma, nu=nu, **kwargs)
