import numpy as np
import cv2 as cv
from F01 import Feature


class F09(Feature):

    def get_map_float32(self, img, roi):
        map_uint8 = self.get_map(img, roi)
        return np.float32(map_uint8) / 255.

    def get_map(self, img, roi):

        # Convert to gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Sobel
        k = 7
        scale = 1.0
        sx = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=k, scale=scale)
        sy = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=k, scale=scale)
        s = np.abs(sx) + np.abs(sy)
        s = self.norm(s, 0, 1e5)
        s = np.uint8(s * 255)
        s = cv.bitwise_and(s, roi)

        # Blur rate
        br = self.norm_scalar(np.max(s), 100, 200)

        res = np.ones_like(gray) * int(br * 255)
        return res
