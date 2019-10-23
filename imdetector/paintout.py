import numpy as np
import cv2 as cv
from .base import BaseDetector, DrawFlags
from .image import SuspiciousImage


class PaintOut(BaseDetector):
    """
    Parameters
    ----------
    ksize : int, (default=11)
    min_area : int, (default=100)
    color : Tuple[int, int, int], (default=(0,255,255))
    flags : int, (default=DrawFlags.SHOW_RESULT)
        See base.DrawFlags

    Attributes
    ----------
    image_ : array,
    ratio_ : float,
    """
    def __init__(self,
                 ksize=11, min_area=100,
                 color=(0, 255, 255), flags=DrawFlags.SHOW_RESULT):
        self.ksize = ksize
        self.min_area = min_area
        self.color = color
        self.flags = flags

    def detect(self, img):
        """
        :param img: SuspiciousImage class instance
        :return: Suspect(1) or not(0)
        :rtype: int
        """
        self.find_flat_area(img)
        result = 1 if hasattr(self, 'ratio_') else 0
        return result

    def find_flat_area(self, img):
        lap = abs(cv.filter2D(
            img.gray, -1, np.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]], np.float32), delta=100).astype('int') - 100)
        lap = np.where(lap < 1, 255, 0).astype('uint8')
        lap = cv.medianBlur(lap, ksize=self.ksize)

        _, contours, hierarchy = cv.findContours(
            lap, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas = np.array([cv.contourArea(cont) for cont in contours])
        contours = np.array(contours)
        cnt = contours[areas > self.min_area]

        if len(cnt) > 0:
            if self.flags == DrawFlags.SHOW_RESULT or self.flags == DrawFlags.SHOW_FULL_RESULT:
                self.image_ = cv.drawContours(img.mat, cnt, -1, self.color, -1)
            self.ratio_ = np.sum(self.image_[:, :, 1] == 255) / img.gray.flatten().shape[0]

        return self
