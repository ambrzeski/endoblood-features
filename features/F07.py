import numpy as np
import cv2 as cv
from F01 import Feature


class F07(Feature):

    def get_map_float32(self, img, img_f01):
        map_uint8 = self.get_map(img, img_f01)
        return np.float32(map_uint8) / 255.

    def get_map(self, img, img_f01):

        # Calculate blur
        k = 55
        blur = cv.blur(img_f01, (k, k))

        # Calculate "spread"
        spread = 1.0 * cv.countNonZero(blur) / (cv.countNonZero(img_f01) or 1)
        spread = self.norm_scalar(spread, 2, 8)

        res = np.uint8(img_f01 * spread)
        return res
