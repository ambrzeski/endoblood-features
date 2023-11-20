import numpy as np


class Feature(object):

    def get_map(self, img, roi):
        map_float32 = self.get_map_float32(img, roi)
        return np.uint8(255 * map_float32)

    def get_map_float32(self, img, roi):
        raise NotImplementedError()

    @staticmethod
    def norm(img, a, b):
        res = img.copy()
        res[res > b] = b
        res[res < a] = a
        res -= a
        res *= 1. / (b - a)
        return res

    @staticmethod
    def norm_scalar(val, a, b):
        res = val
        if res > b:
            res = b
        if res < a:
            res = a
        res -= a
        res *= 1. / (b - a)
        return res


class F01(Feature):

    c1_norm = (0.003, 0.01)
    norm_thresh_a = 0.2
    norm_thresh_b = 1.1

    def get_map_float32(self, img, roi):
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Prevent div by 0
        b[b == 0] = 1
        g[g == 0] = 1

        # Calculate "redness" coefficients
        c1 = np.minimum(r / (1.0 * b), r / (1.0 * g)) / 255.
        c1 -= 1. / 255.
        c1[c1 < 0] = 0
        c1 = self.norm(c1, *self.c1_norm)
        c2 = r / 255.
        c3 = (255. - np.abs(1.0 * b - g)) / 255.

        # Calculate "redness" map
        red = np.power(c1, 1) * np.power(c2, 0) * np.power(c3, 5)
        red = self.norm(red, self.norm_thresh_a, self.norm_thresh_b)
        red[r < 30] = 0

        # Cut off the area outside ROI
        red[roi == 0] = 0

        return red
