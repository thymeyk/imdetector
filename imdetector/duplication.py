import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from .base import BaseDetector, DrawFlags
from .image import Image


class Duplication(BaseDetector):
    """
    Parameters
    ----------
    r : float, (default=0.60)
    min_match : int, (default=5)
    min_key : float, (default=0.5)
    ransacT : float, (default=5.0)
    crossCheck : bool, (default=False)
    color : Tuple[int, int, int], (default=(0,255,255))
    flags : int, (default=DrawFlags.SHOW_RESULT)
        See base.DrawFlags

    Attributes
    ----------
    matches_ : list[cv.DMatch],
    mask_ : array,
    M_ : array,
    image_ : array,
    """

    def __init__(self, r=0.60, min_match=5, min_key=0.5, ransacT=5.0, crossCheck=False, color=(0,255,255), flags=DrawFlags.SHOW_RESULT):
        self.r = r
        self.min_match = min_match
        self.min_key = min_key
        self.ransacT = ransacT
        self.crossCheck = crossCheck
        self.color = color
        self.flags = flags


    def detect(self, img1: Image, img2: Image) -> int:
        """
        :param img1: Image class instance
        :param img2: Image class instance
        :return: int, detect(1) or not(0)
        """
        if (len(img1.kp) < 2 or len(img2.kp) < 2):
            return 0

        if self.crossCheck:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            self.matches_ = bf.match(img1.des, img2.des)
        else:
            matches = img1.bf.knnMatch(img1.des, img2.des, k=2)
            self.matches_ = self.ratio_test(matches)

        self.result_path_ = self.find_homography(img1, img2)
        result = 1 if hasattr(self, 'mask_') else 0

        return result


    def ratio_test(self, matches: list) -> list:
        """
        1位のマッチのL2ノルムが2位のマッチのL2ノルムの(r*100)%以下のマッチを残す

        :param matches: list of the first and the second cv.DMatch
        :return: list of cv.DMatch
        """
        good_matches = []
        for m, n in matches:
            if m.distance <= self.r * n.distance and n.distance != 0:
                good_matches.append(m)
        return good_matches


    def find_homography(self, img1: Image, img2: Image):
        """
        この関数は２つの画像から得られた点の集合を与えると，その物体の射影変換を計算する。
        ransacReprojThresholdは点の組をインライア値として扱うために許容される逆投影誤差の最大値。1~10が妥当。
        Mは3*3の射影変換行列

        :param img1: Image class instance
        :param img2: Image class instance
        :return:
        """
        if len(self.matches_) > self.min_match:
            src_pts = np.float32([img1.kp[m.queryIdx].pt for m in self.matches_]
                                 ).reshape(-1, 1, 2)  # image1のkeypointの座標 (n,1,2)
            dst_pts = np.float32([img2.kp[m.trainIdx].pt for m in self.matches_]
                                 ).reshape(-1, 1, 2)  # image2のkeypointの座標 (n,1,2)

            self.M_, self.mask_ = cv.findHomography(
                src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=self.ransacT)
            matches_mask = self.mask_.ravel().tolist()

            # キーポイントのうちminKey以上を含むならマスクを採用（＝DETECT!）
            if matches_mask.count(1) / len(matches_mask) >= self.min_key:
                if DrawFlags(self.flags) != DrawFlags.RETURN_RESULT:
                    self.draw_match(img1, img2, src_pts)
        return self


    def draw_match(self, img1: Image, img2: Image, src_pts: np.ndarray):
        x, y, w, h = cv.boundingRect(
            np.float32(src_pts)[self.mask_.ravel() == 1])
        img1_rect = cv.rectangle(
            img1.mat.copy(), (x, y), (x + w, y + h), self.color, 2)

        pts = np.float32([[x, y], [x, y + h], [x + w, y + h], [x + w, y]]
                         ).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, self.M_)

        img2_rect = cv.polylines(img2.mat.copy(), [np.int32(dst)], isClosed=True, color=self.color, thickness=3, lineType=cv.LINE_AA)  # LINE_AA:アンチエイリアス

        if DrawFlags(self.flags) == DrawFlags.SHOW_RESULT:
            img1_name, img2_name = img1.name.split('/')[-1], img2.name.split('/')[-1]

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.imshow(cv.cvtColor(img1_rect, cv.COLOR_BGR2RGB))
            ax1.set_title(img1_name)
            ax1.axis('off')

            ax2 = fig.add_subplot(122)
            ax2.imshow(cv.cvtColor(img2_rect, cv.COLOR_BGR2RGB))
            ax2.set_title(img2_name)
            ax2.axis('off')

            fig.tight_layout(pad=0)
            fig.canvas.draw()
            s, (width, height) = fig.canvas.print_to_buffer()
            X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            self.image_ = cv.cvtColor(X, cv.COLOR_RGBA2BGRA)

            plt.close()

        elif DrawFlags(self.flags) == DrawFlags.SHOW_FULL_RESULT:
            draw_params = dict(matchColor=self.color,
                               singlePointColor=None,
                               matchesMask=self.mask_.ravel().tolist(),  # draw only inliers
                               flags=2)
            self.image_ = cv.drawMatches(
                img1_rect, img1.kp, img2_rect, img2.kp, self.matches_, None, **draw_params)

        return self


    def save_image(self, filename: str):
        if hasattr(self, 'image_'):
            cv.imwrite(filename, self.image_)
        else:
            print('Error: There is no result.')
        return self


class CopyMove(Duplication):
    """
    Parameters
    ----------
    r : float, (default=0.60)
    min_dis : int, (default=40)
    min_match : int, (default=5)
    min_key : float, (default=0.5)
    ransacT : float, (default=5.0)
    crossCheck : bool, (default=False)
    color : Tuple[int, int, int], (default=(0,255,255))
    flags : int, (default=DrawFlags.SHOW_RESULT)
        See base.DrawFlags

    Attributes
    ----------
    matches_ : list[cv.DMatch],
    mask_ : array,
    M_ : array,
    image_ : array,
    """

    def __init__(self, r=0.60, min_dis=40, min_match=5, min_key=0.5, ransacT=5.0, crossCheck=False, color=(0,255,255), flags=DrawFlags.SHOW_RESULT):
        super(CopyMove, self).__init__(r, min_match, min_key, ransacT, crossCheck, color, flags)
        self.min_dis = min_dis


    def detect(self, img1: Image, img2=None) -> int:
        """
        :param img1: Image class instance
        :return: int, detect(1) or not(0)
        """
        if len(img1.kp) < 3:
            return 0

        matches3 = img1.bf.knnMatch(img1.des, img1.des, k=3)

        # 自分自身にマッチしているものを除く
        matches2 = []
        for m in matches3:
            if len(m) > 1:
                m = [mi for mi in m if img1.kp[mi.queryIdx] != img1.kp[mi.trainIdx]]
                if len(m) > 1:
                    matches2.append([m[0], m[1]])
            else:
                return 0

        good = self.ratio_test(matches2)
        self.matches_ = self.distance_cutoff(img1.kp, good)

        self.result_path_ = self.find_homography(img1)
        result = 1 if hasattr(self, 'mask_') else 0

        return result


    def distance_cutoff(self, kp: list, matches: list) -> list:
        """
        マッチの重複（相互）とマッチ間の距離がminDis以下のマッチを除く

        :param kp: list of cv.KeyPoint
        :param matches: list of cv.DMatch
        :return: better list of cv.DMatch
        """
        for m in matches:
            if kp[m.queryIdx].pt < kp[m.trainIdx].pt:
                m.queryIdx, m.trainIdx = m.trainIdx, m.queryIdx
        better = [
            m
            for m in matches
            if (np.linalg.norm(np.array(kp[m.queryIdx].pt) - np.array(kp[m.trainIdx].pt)) > self.min_dis)
        ]
        return better


    def find_homography(self, img1: Image, img2=None):
        super(CopyMove, self).find_homography(img1=img1, img2=img1)


    def draw_match(self, img1: Image, img2: Image, src_pts: np.ndarray):
        x, y, w, h = cv.boundingRect(
            np.float32(src_pts)[self.mask_.ravel() == 1])
        img_rect = cv.rectangle(
            img1.mat.copy(), (x, y), (x + w, y + h), self.color, 2)

        pts = np.float32([[x, y], [x, y + h], [x + w, y + h], [x + w, y]]
                         ).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, self.M_)

        self.image_ = cv.polylines(img_rect, [np.int32(dst)], isClosed=True,
                                   color=self.color, thickness=3, lineType=cv.LINE_AA)

        if DrawFlags(self.flags) == DrawFlags.SHOW_FULL_RESULT:
            for m in self.matches_:
                cv.line(self.image_, tuple(map(round, img1.kp[m.queryIdx].pt)), tuple(
                    map(round, img1.kp[m.trainIdx].pt)), color=self.color)
                cv.circle(self.image_, tuple(map(round, img1.kp[m.queryIdx].pt)), 3,
                          color=self.color, thickness=2)
                cv.circle(self.image_, tuple(map(round, img1.kp[m.trainIdx].pt)), 3,
                          color=self.color, thickness=2)

        return self

