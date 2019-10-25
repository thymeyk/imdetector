import os
import unittest

from imdetector.image import SuspiciousImage
from imdetector.cutpaste import CutPaste

DIR = os.getcwd()


class TestCutPaste(unittest.TestCase):
    def test_noise(self):
        expected = [1, 0]
        img1 = SuspiciousImage(
            os.path.join(
                DIR, 'test/image/yrc_1000_505_cp.png'))
        img2 = SuspiciousImage(os.path.join(DIR, 'test/image/yrc_16.png'))
        detector = CutPaste()
        actual = detector.detect([img1, img2])
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])


if __name__ == "__main__":
    unittest.main()
