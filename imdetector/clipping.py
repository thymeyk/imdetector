import numpy as np
import cv2 as cv
from .base import BaseDetector, DrawFlags


class Clipping(BaseDetector):
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
        self.ratio_ = 0
        self.image_ = None
        self.find_flat_area(img)
        result = 1 if self.ratio_ > 0 else 0
        return result

    def find_flat_area(self, img):
        lap = img.lap
        lap = np.where(lap < 1, 255, 0).astype('uint8')
        lap = cv.medianBlur(lap, ksize=self.ksize)
        white = np.where(img.gray == 255, 0, 255).astype('uint8')
        lap = lap & white

        _, contours, hierarchy = cv.findContours(
            lap, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas = np.array([cv.contourArea(cont) for cont in contours])
        contours = np.array(contours)
        cnt = contours[areas > self.min_area]

        if len(cnt) > 0:
            if self.flags == DrawFlags.SHOW_RESULT or self.flags == DrawFlags.SHOW_FULL_RESULT:
                self.image_ = cv.drawContours(
                    img.mat.copy(), cnt, -1, self.color, -1)
            self.ratio_ = np.sum(lap == 255) / img.gray.flatten().shape[0]

        return self
