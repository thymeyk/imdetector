import os
import unittest

from imdetector.image import SuspiciousImage
from imdetector.clipping import Clipping

DIR = os.getcwd()


class TestClipping(unittest.TestCase):
    def test_paintout1(self):
        expected = 1
        img1 = SuspiciousImage(os.path.join(DIR, 'test/image/yrc_5_po.png'))
        detector = Clipping()
        actual = detector.detect(img1)
        detector.save_image(
            os.path.join(
                DIR, 'test/image/paintout_result.jpg'))
        self.assertEqual(expected, actual)

    def test_paintout0(self):
        expected = 0
        img1 = SuspiciousImage(os.path.join(DIR, 'test/image/yrc_16.png'))
        detector = Clipping()
        actual = detector.detect(img1)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
