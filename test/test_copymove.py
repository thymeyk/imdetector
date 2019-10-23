import unittest

from imdetector.copymove import CopyMove, Duplication
from imdetector.image import SuspiciousImage


class TestDuplication(unittest.TestCase):
    def test_duplication(self):
        expected = 1
        img1 = SuspiciousImage('./image/yrc_16.png')
        img2 = SuspiciousImage('./image/yrc_16_60degree.png')
        detector = Duplication()
        actual = detector.detect(img1, img2)
        detector.save_image('./image/duplication_result.jpg')
        self.assertEqual(expected, actual)

    def test_duplication2(self):
        expected = 0
        img1 = SuspiciousImage('./image/yrc_16.png')
        img2 = SuspiciousImage('./image/yrc_5_inpaint.png')
        detector = Duplication()
        actual = detector.detect(img1, img2)
        self.assertEqual(expected, actual)


class TestCopyMove(unittest.TestCase):
    def test_copymove(self):
        expected = 1
        img1 = SuspiciousImage('./image/yrc_7_cm.png')
        detector = CopyMove()
        actual = detector.detect(img1)
        detector.save_image('./image/copymove_result.jpg')
        self.assertEqual(expected, actual)

    def test_copymove2(self):
        expected = 0
        img1 = SuspiciousImage('./image/yrc_16.png')
        detector = CopyMove()
        actual = detector.detect(img1)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
