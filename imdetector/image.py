import os

import cv2 as cv
import numpy as np


class SuspiciousImage:
    def __init__(self,
                 path=None, hist_eq=True,
                 algorithm='orb', nfeatures=5000,
                 dsize=256, gap=32, h=200, w=400):
        self.path = path
        self.hist_eq = hist_eq
        self.algorithm = algorithm
        self.nfeatures = nfeatures
        self.dsize = dsize
        self.gap = gap
        self.h = h
        self.w = w

        self.mat = None
        self.gray = None

        if path is not None:
            self.name = os.path.basename(path)
            self.size = os.path.getsize(path)
            self.read()

    def read(self):
        # Read image
        if self.path.split('.')[-1] == 'gif':
            gif = cv.VideoCapture(self.path)
            _, self.mat = gif.read()
        else:
            self.mat = cv.imread(self.path)
        if self.mat is None:
            print("ERROR: No such image file.")
            return self

        # Convert to Gray image
        if len(self.mat.shape) is 3:
            self.gray = cv.cvtColor(self.mat, cv.COLOR_BGR2GRAY)
        elif len(self.mat.shape) is 2:
            self.gray = self.mat

        self.keypoint()
        self.laplacian()

        # 保存用のサイズ
        height, width = self.gray.shape
        scale = self.h / height
        if width * scale > self.w:
            scale = self.w / width
        self.h = int(height * scale)
        self.w = int(width * scale)

        self.noise = 0
        self.no_img = cv.resize(
            255 -
            np.where(
                self.lap > 4,
                4,
                self.lap) /
            4 *
            255,
            dsize=(
                self.w,
                self.h))
        self.dist = 0

        self.clipping = 0
        # self.cl_img = self.mat
        self.area_ratio = 0

        self.copymove = -1
        # self.cm_img = self.mat
        self.mask_ratio = 0

        self.cutpaste = -1
        self.prob = 0

        return self

    def make_flip(self, imgarr, name):
        self.name = name
        self.mat = cv.flip(imgarr, 1)
        self.gray = cv.cvtColor(self.mat, cv.COLOR_BGR2GRAY)
        self.laplacian()
        self.keypoint()
        return self

    def laplacian(self):
        self.lap = abs(cv.filter2D(
            self.gray, -1,
            np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]], np.float32), delta=100).astype('int') - 100)
        return self

    def keypoint(self):
        if self.mat is None:
            self.mat = cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR)
        elif self.gray is None:
            self.gray = cv.cvtColor(self.mat, cv.COLOR_BGR2GRAY)

        keyimg = self.gray

        if self.hist_eq:
            keyimg = cv.equalizeHist(keyimg)

        H, W = keyimg.shape
        gap = self.gap
        imgzeros = np.full((H + gap * 2, W + gap * 2), 255)
        for i in range(H):
            for j in range(W):
                imgzeros[i + gap, j + gap] = keyimg[i, j]
        self.keyimg = imgzeros.astype('uint8')

        if self.algorithm == 'orb':
            self.detector = cv.ORB_create(nfeatures=self.nfeatures)
            self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        elif self.algorithm == 'akaze':
            self.detector = cv.AKAZE_create()
            self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        elif self.algorithm == 'sift':
            self.detector = cv.xfeatures2d.SIFT_create()
            self.bf = cv.BFMatcher(cv.NORM_L2)
        elif self.algorithm == 'surf':
            self.detector = cv.xfeatures2d.SURF_create()
            self.bf = cv.BFMatcher(cv.NORM_L2)

        self.kp, self.des = self.detector.detectAndCompute(self.keyimg, None)

        return self
