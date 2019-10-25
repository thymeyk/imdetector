import os
import unittest

from imdetector.image import SuspiciousImage
from imdetector.noise import Noise

DIR = os.getcwd()


class TestNoise(unittest.TestCase):
    def test_noise(self):
        expected = [1, 0]
        img1 = SuspiciousImage(os.path.join(DIR, 'test/image/yrc_1_no.png'))
        img2 = SuspiciousImage(os.path.join(DIR, 'test/image/yrc_7_cm.png'))
        detector = Noise()
        actual = detector.detect([img1, img2])
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])


if __name__ == "__main__":
    unittest.main()
