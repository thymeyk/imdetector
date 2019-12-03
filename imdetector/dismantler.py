import os
import cv2 as cv
import numpy as np
import seaborn as sns

sns.set(style='white', context='talk')


class Dismantler:
    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = {
                'blackThres': 0.8,
                'whiteThres': 0.8,
                'white2Thres': 0.001,
                'whiteblackThres': 0.9,
                'areaThres': 1000,
                'splitThres': 0.8,
                'varThres': 3,
                'var2Thres': 50}
        self.thresholds = thresholds

    def dismantle(self, files, SAVE_DIR):
        for path in files:
            img = cv.imread(path)
            basename = os.path.basename(path)  # oo.jpg
            subimgs = self.get_sub_images(img)
            if subimgs is None:
                continue
            for i, subimg in enumerate(subimgs):
                subimg_DIR = os.path.join(SAVE_DIR, 'subimg')
                os.makedirs(subimg_DIR, exist_ok=True)
                savename = os.path.join(
                    subimg_DIR, '{}-{:03}.png'.format(basename, i))
                cv.imwrite(savename, subimg)
                subimg_cut_DIR = os.path.join(SAVE_DIR, 'subimg_cut')
                os.makedirs(subimg_cut_DIR, exist_ok=True)
                savename = os.path.join(
                    subimg_cut_DIR, os.path.basename(savename))
                self.get_rectangle(subimg, savename)

    @staticmethod
    def indices(a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def get_fire_lane_map(self, gray, orientation=0, offset=None):
        # Pre-processing #
        fire_lane_map = np.zeros(gray.shape[0:2])
        imgDim = gray.shape
        arraySum = np.sum(gray, axis=orientation)
        arraySum_nor = arraySum / float(np.max(arraySum))
        arrayVar = np.var(abs(gray - 127), axis=orientation)

        blank_line = self.indices(zip(arrayVar, arraySum_nor), lambda x: x[0] < self.thresholds['varThres'] or (
            x[0] < self.thresholds['var2Thres'] and x[1] > self.thresholds['splitThres']))

        # Alternate orientation #
        if offset is None:
            offset = [0, 0]
        if orientation is 1:
            if len(blank_line) > 0:
                blank_line_loc = np.asarray(blank_line) + offset[0]
                fire_lane_map[blank_line_loc,
                              offset[1]: offset[1] + imgDim[1]] = 1
        else:
            if len(blank_line) > 0:
                blank_line_loc = np.asarray(blank_line) + offset[1]
                fire_lane_map[offset[0]: offset[0] +
                              imgDim[0], blank_line_loc] = 1

        return fire_lane_map

    def get_effective_region_mask(self, gray):
        fire_lane_map_vertical = self.get_fire_lane_map(gray, orientation=0)
        fire_lane_map_horizontal = self.get_fire_lane_map(gray, orientation=1)
        mask = fire_lane_map_vertical + fire_lane_map_horizontal
        mask = np.where(mask > 0, 0, 1)
        return mask

    def get_sub_contours(self, gray, offset=None):
        if offset is None:
            offset = [0, 0]
        area = gray.flatten().shape[0]
        white_ratio = np.sum(gray == 255) / area
        black_ratio = np.sum(gray == 0) / area

        if (white_ratio > self.thresholds['whiteThres'] or black_ratio > self.thresholds['blackThres']) or (
                black_ratio + white_ratio > self.thresholds['whiteblackThres'] or area < self.thresholds['areaThres']):
            return None
        if white_ratio < self.thresholds['white2Thres']:
            return [np.array([[[0, 0]], [[0, gray.shape[0]]], [[gray.shape[1], gray.shape[0]]], [
                [gray.shape[1], 0]]], dtype='int32') + offset]

        mask = self.get_effective_region_mask(gray)
        _, contours, hierarchy = cv.findContours(
            (mask * 255).astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(cont) for cont in contours]
        cnts = [contours[i]
                for i in range(len(contours)) if areas[i] > max(areas) / 10]
        bound = [cv.boundingRect(cnt) for cnt in cnts]
        if len(bound) is 1:
            return [cnts[0] + offset]

        subcnts = []
        for x, y, w, h in bound:
            subsubcnts = self.get_sub_contours(
                gray[y:y + h, x:x + w], offset=[x, y])
            if subsubcnts is not None:
                subcnts += subsubcnts

        return [c + offset for c in subcnts]

    def get_sub_images(self, img):
        """
        parameters
        ----------
        img: numpy.ndarray (uint8)
        """
        if len(img.shape) is 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        self.subcnts_ = self.get_sub_contours(gray)
        if self.subcnts_ is None:
            return None
        bound = [cv.boundingRect(cnt) for cnt in self.subcnts_]
        subimgs = [img[y: y + h, x: x + w] for x, y, w, h in bound]
        return subimgs

    @staticmethod
    def get_rectangle(img, savename):
        """
        parameters
        ----------
        img: numpy.ndarray (uint8)
        savename: string
        """
        # Otsu threshold
        if len(img.shape) is 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
        retval, dst = cv.threshold(
            gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        dst = cv.bitwise_not(dst)

        # get rectangle candidate
        image, contours, hierarchy = cv.findContours(
            dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(cont) for cont in contours]
        rect = [contours[i] for i in range(len(areas)) if areas[i] > 10000]
        rect = [cnt for cnt in rect if len(cnt) < 50]

        # get bound line
        bound = [cv.boundingRect(cnt) for cnt in rect]

        # draw filled rectangle
        img_rect = np.zeros(gray.shape).astype('uint8')
        for x, y, w, h in bound:
            cv.rectangle(img_rect, (x, y), (x + w, y + h), (255, 255, 255), -1)

        # get contours again
        image, contours, hierarchy = cv.findContours(
            img_rect, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bound = [cv.boundingRect(cnt) for cnt in contours]
        for i, (x, y, w, h) in enumerate(bound):
            if len(img.shape) is 3:
                dst = img[y: y + h, x: x + w, :]
            else:
                dst = img[y: y + h, x: x + w]
            if img.shape[0] > 20 and img.shape[1] > 20:
                print('{}-{:03}.png'.format(savename.rsplit('.', 1)[-2], i))
                cv.imwrite('{}-{:03}.png'.format(savename.rsplit('.', 1)
                                                 [-2], i), np.array(dst).astype('uint8'))
            # cvimshow(np.array(dst).astype('uint8'))
        if len(bound) is 0:
            if img.shape[0] * \
                    img.shape[1] > 1000 and img.shape[0] > 20 and img.shape[1] > 20:
                cv.imwrite('{}-{:03}.png'.format(savename.rsplit('.', 1)
                                                 [-2], 0), img)
