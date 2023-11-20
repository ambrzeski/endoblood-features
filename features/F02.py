import numpy as np
import cv2 as cv
from F01 import Feature


class F02(Feature):

    def get_map(self, img, img_f01, eroded_roi):
        map_float32 = self.get_map_float32(img, img_f01, eroded_roi, normalize=False)
        return np.uint8(map_float32)

    def get_map_float32(self, img, img_f01, eroded_roi, normalize=True):

        # Get inverted S (HSV) channel
        hsv_s = 255 - cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 1]
        hsv_s = cv.blur(hsv_s, (7, 7))

        # Compute Sobel
        k = 7
        scale = 1.0
        depth = cv.CV_32F
        sx = cv.Sobel(hsv_s, ddepth=depth, dx=1, dy=0, ksize=k, scale=scale)
        sy = cv.Sobel(hsv_s, ddepth=depth, dx=0, dy=1, ksize=k, scale=scale)

        # Add up
        s = np.abs(sx) + np.abs(sy)
        s = self.norm(s, 0., 1e5)

        # Clear sobel response outside ROI
        s = 1 - s
        s[eroded_roi == 0] = 0

        # Combine with F01
        result = s * img_f01

        # Normalize result, if requested
        if normalize:
            result /= 255.

        return result
