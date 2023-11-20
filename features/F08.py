import numpy as np
from F01 import Feature


class F08(Feature):

    def get_map_float32(self, img, roi):
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Prevent div by 0
        b[b == 0] = 1
        g[g == 0] = 1

        # Prepare "redness" map
        red = np.minimum(r / (1.0 * b), r / (1.0 * g))
        red /= 255.
        red = self.norm(red, 0.005, 0.01)

        # Suppress "redness" map with total image redness - apply penalty for red dominated images
        total_red = np.sum(red) / (img.shape[0] * img.shape[1])
        total_red = np.ones_like(red) * total_red
        total_red = self.norm(total_red, 0.1, 0.50)

        result = red * (1 - total_red)
        result[roi == 0] = 0
        return result
