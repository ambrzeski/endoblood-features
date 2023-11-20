import F02


class F05(F02.F02):

    def get_map(self, img, img_f04, eroded_roi):
        return super(F05, self).get_map(img, img_f04, eroded_roi)

    def get_map_float32(self, img, img_f04, eroded_roi, normalize=True):
        return super(F05, self).get_map_float32(img, img_f04, eroded_roi, normalize)
