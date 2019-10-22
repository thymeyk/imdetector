import cv2 as cv
import numpy as np


class SuspectImage():
    def __init__(self, name=None, isGabor=False, isHist=False, algorithm='orb', nfeatures=2000):
        self.name = ''
        self.mat = 0
        self.gray = 0
        self.isHist = isHist
        self.isGabor = isGabor

        if name is not None:
            self.read(name, algorithm, nfeatures)

    def read(self, name, algorithm, nfeatures):
        # Read image
        self.name = name
        if name.split('.')[-1] == 'gif':
            gif = cv.VideoCapture(self.name)
            _, self.mat = gif.read()
        else:
            self.mat = cv.imread(self.name)

        # Convert to Gray image
        self.gray = cv.cvtColor(self.mat, cv.COLOR_BGR2GRAY)

        self.keypoint(algorithm=algorithm, nfeatures=nfeatures)

    def keypoint(self, algorithm='orb', nfeatures=2000):
        if type(self.mat) == type(0):
            self.mat = cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR)
        elif type(self.gray) == type(0):
            self.gray = cv.cvtColor(self.mat, cv.COLOR_BGR2GRAY)

        self.keyimg = self.gray

        if self.isGabor:
            sigma = 8
            filtered = [cv.filter2D(self.keyimg, -1,
                                    cv.getGaborKernel(ksize=(4 * sigma, 4 * sigma),
                                                      sigma=sigma,
                                                      theta=np.radians(
                                                          i * 22.5),
                                                      lambd=10,
                                                      gamma=2,
                                                      psi=0))
                        for i in range(8)]
            self.keyimg = np.mean(np.array(filtered), axis=0).astype('uint8')
        if self.isHist:
            self.keyimg = cv.equalizeHist(self.keyimg)

        if algorithm == 'orb':
            self.detector = cv.ORB_create(nfeatures=nfeatures)
            self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        elif algorithm == 'akaze':
            self.detector = cv.AKAZE_create()
            self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        elif algorithm == 'sift':
            self.detector = cv.xfeatures2d.SIFT_create()
            self.bf = cv.BFMatcher(cv.NORM_L2)
        elif algorithm == 'surf':
            self.detector = cv.xfeatures2d.SURF_create()
            self.bf = cv.BFMatcher(cv.NORM_L2)

        self.kp, self.des = self.detector.detectAndCompute(self.keyimg, None)
