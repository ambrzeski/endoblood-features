import F03


class F06(F03.F03):

    def get_map(self, img, img_f04, eroded_roi):
        return super(F06, self).get_map(img, img_f04, eroded_roi)

    def get_map_float32(self, img, img_f04, eroded_roi, normalize=True):
        return super(F06, self).get_map_float32(img, img_f04, eroded_roi, normalize)
