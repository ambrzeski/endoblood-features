import cv2 as cv
import F01, F02, F03, F04, F05, F06, F07, F08, F09


class Features(object):

    def __init__(self):
        self.features = {
            "F01": F01.F01(),
            "F02": F02.F02(),
            "F03": F03.F03(),
            "F04": F04.F04(),
            "F05": F05.F05(),
            "F06": F06.F06(),
            "F07": F07.F07(),
            "F08": F08.F08(),
            "F09": F09.F09()
        }
        self.names = sorted(self.features.keys())

    def get(self, img, roi):
        # Erode ROI
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        eroded_roi = cv.erode(roi, kernel, iterations=1)

        # Compute feature maps
        f01_map = self.features["F01"].get_map(img, roi)
        f02_map = self.features["F02"].get_map(img, f01_map, eroded_roi)
        f03_map = self.features["F03"].get_map(img, f01_map, eroded_roi)
        f04_map = self.features["F04"].get_map(img, roi)
        f05_map = self.features["F05"].get_map(img, f04_map, eroded_roi)
        f06_map = self.features["F06"].get_map(img, f04_map, eroded_roi)
        f07_map = self.features["F07"].get_map(img, f01_map)
        f08_map = self.features["F08"].get_map(img, roi)
        f09_map = self.features["F09"].get_map(img, eroded_roi)
        features = [
            f01_map,
            f02_map,
            f03_map,
            f04_map,
            f05_map,
            f06_map,
            f07_map,
            f08_map,
            f09_map
        ]

        return features
