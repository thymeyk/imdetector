import numpy as np
import cv2 as cv
from .base import BaseDetector, DrawFlags
from .image import SuspectImage


class PaintOut(BaseDetector):
    """
    Parameters
    ----------

    Attrobites
    ----------
    image_ : array,
    ratio_ : float,
    """
    def __init__(self, flags=DrawFlags.SHOW_RESULT):
        self.flags = flags


    def detect(self, img: SuspectImage) -> int:
        self.find_flat_area(img)
        result = 1 if hasattr(self, 'ratio_') else 0
        return result


    def find_flat_area(self, img: SuspectImage):
        lap = abs(cv.filter2D(
            img.gray, -1, np.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]], np.float32), delta=100).astype('int') - 100)
        lap = np.where(lap < 1, 255, 0).astype('uint8')
        lap = cv.medianBlur(lap, 11)

        _, contours, hierarchy = cv.findContours(
            lap, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas = np.array([cv.contourArea(cont) for cont in contours])
        contours = np.array(contours)
        cnt = contours[areas > 100]

        if len(cnt) > 0:
            if self.flags == DrawFlags.SHOW_RESULT or self.flasgs == DrawFlags.SHOW_FULL_RESULT:
                self.image_ = cv.drawContours(img.mat, cnt, -1, (0, 255, 255), -1)
            self.ratio_ = np.sum(self.image_[:, :, 1] == 255) / img.gray.flatten().shape[0]

        return self
