import colorsys
from numpy import histogram, asarray, resize
from skimage.io import imsave
from predict.featureextractor import FeatureExtractor

__author__ = 'Group 16'

"""

    This class contains an extractor that extracts colour information. It converts a hue image from a color image and
    calculate a histogram of these values.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""


class ColorFeatureExtractor(FeatureExtractor):

    def __init__(self, _bins):
        self.bins = _bins

    def extract_feature_vector(self, image):
        """
        Extract a hue color histogram from image
        :param image: a COLOR image
        :return: a histogram of the HUE image which is converted from the input COLOR image
        """
        hue = ColorFeatureExtractor.extract_hue(image)
        return self.calculate_histogram(hue, self.bins)

    def calculate_histogram(self, hue, bins=20):
        hist = histogram(hue, bins=bins, range=(0, 1))

        return [x / sum(hist[0]) for x in hist[0]]

    @staticmethod
    def extract_hue(img, binary=False, debug=False):

        # Converting the RGB values to HSV values
        hsv = [None] * len(img)
        for i in range(len(img)):
            temp = [None] * len(img[0])
            for j in range(len(img[0])):
                temp[j] = colorsys.rgb_to_hsv(img[i, j, 0], img[i, j, 1], img[i, j, 2])
            hsv[i] = temp

        # Extracting the 3 channels
        hue = [None] * len(img)
        saturation = [None] * len(img)
        value = [None] * len(img)
        for x in range(len(img)):
            hue_temp = [None] * len(img[0])
            saturation_temp = [None] * len(img[0])
            value_temp = [None] * len(img[0])
            for y in range(len(img[0])):
                hue_temp[y] = hsv[x][y][0]
                saturation_temp[y] = hsv[x][y][1]
                value_temp[y] = hsv[x][y][2]
            hue[x] = hue_temp
            saturation[x] = saturation_temp
            value[x] = value_temp

        # Normalize the Value values
        value_norm = [None] * len(value)
        for x in range(len(value)):
            value_norm_temp = [None] * len(value[x])
            for y in range(len(value[x])):
                value_norm_temp[y] = value[x][y] / 255
            value_norm[x] = value_norm_temp
        value = value_norm

        # We now loop through the hue values and filter out the ones with red values and set them to white
        # To avoid inconsistencies, we need to make sure the hue value lays in his chromatic area,
        # else we set the pixel to black
        for x in range(len(hue)):
            for y in range(len(hue[x])):
                if (hue[x][y] > 0.95 and hue[x][y] <= 1) or (hue[x][y] >= 0 and hue[x][y] < 0.05):
                    hue[x][y] = 1
                elif binary:
                    hue[x][y] = 0

                if saturation[x][y] < 0.25:  # Achromatic area
                    hue[x][y] = 0
                if value[x][y] < 0.2 or value[x][y] > 0.9:  # Achromatic area
                    hue[x][y] = 0

        if debug:
            imsave('test.png', asarray(hue))

        return hue
