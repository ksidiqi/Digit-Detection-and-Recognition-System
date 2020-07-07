from unittest import TestCase
import cv2
from MultiDigitRecognition import Util

class TestUtil(TestCase):
    def test_get_img_pyramid(self):
        img = cv2.imread("./data/tmp.jpg")
        u = Util()
        for i, im in enumerate(u.get_img_pyramid(img)):
            cv2.imshow(str(i), im)
            cv2.waitKey(0)


    def test_de_noise(self):
        self.fail()

    def test_sliding_window(self):
        img = cv2.imread("./data/tmp.jpg")
        u = Util()
        for i, im in enumerate(u.sliding_window(img, size=(128, 128))):
            cv2.imshow(str(i), u.padd_with_zeros(im, (2*im.shape[0], 3*im.shape[1], 3) ))
            cv2.waitKey(0)


    def test_fast_forward(self):
        self.fail()

    def test_sliding_window(self):
        img = cv2.imread("./data/tmp2.jpg")
        u = Util()
        u.cv2_text(img)
