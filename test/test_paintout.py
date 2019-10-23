import unittest

from imdetector.image import SuspiciousImage
from imdetector.paintout import PaintOut


class TestPaintOut(unittest.TestCase):
    def test_copymove(self):
        expected = 1
        img1 = SuspiciousImage('./image/yrc_5_po.png')
        detector = PaintOut()
        actual = detector.detect(img1)
        detector.save_image('./image/paintout_result.jpg')
        self.assertEqual(expected, actual)

    def test_copymove2(self):
        expected = 0
        img1 = SuspiciousImage('./image/yrc_16.png')
        detector = PaintOut()
        actual = detector.detect(img1)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
