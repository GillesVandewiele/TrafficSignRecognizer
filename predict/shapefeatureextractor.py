import cv2
import mahotas
from pylab import *
from skimage import color
from predict.colorfeatureextractor import ColorFeatureExtractor
from predict.featureextractor import FeatureExtractor


__author__ = 'Group16'

"""

    This class will extract moments (geometric and Zernike) from a hue image.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning
"""


class ShapeFeatureExtractor(FeatureExtractor):

    def __init__(self, _radius):
        self.radius = _radius

    def extract_feature_vector(self, image):
        """
        extract geometric and Zernike moments from a hue image which is first converted to a grayscale image
        :param image: a hue image (can be extracted using the Color Feature Extractor
        :return: a feature vector containing geometric and Zernike moments
        """
        # hue_image = ColorFeatureExtractor.extract_hue(image)
        # hue_gray_image = (np.rint(asarray(hue_image) * 255)).astype(np.uint8)
        # contour = self.calculateRimContour(hue_gray_image)
        # feature_vector = self.calculateGeometricMoments(contour)
        # feature_vector = append(feature_vector, self.extract_zernike(color.rgb2gray(image), self.radius))
        feature_vector = self.extract_zernike(color.rgb2gray(image), self.radius)
        return feature_vector

    @staticmethod
    def calculateRimContour(hue):
        img = resize(hue, (64, 64))
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        imgg, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 1)
        if len(contours) == 0:
            return None
        index = 0;
        max_area = cv2.contourArea(contours[0])
        for i in range(0, len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_area = area
                index = i
        return contours[index]

    @staticmethod
    def calculateGeometricMoments(contour):
        if contour is None:
            return [0, 0, 0, 0]

        mu = cv2.moments(contour, False)

        # The zero order central moment is zero so no shape can be predicted
        if (mu["m00"] == 0):
            return [0, 0, 0, 0]

        # Calculate first moment invariant
        I = (mu["mu20"] * mu["mu02"] - mu["mu11"] ** 2) / (mu["m00"] ** 4)

        # Measure ellipticity
        E = 0
        if I < 1 / (16 * math.pi ** 2):
            E = 16 * (math.pi ** 2) * I
        else:
            E = 1 / (16 * (math.pi ** 2) * I)

        # Measure triangularity
        T = 0
        if I < 1 / 108:
            T = 108 * I
        else:
            T = 1 / (108 * I)

        # Measure octagonality
        O = 0
        if I < 1 / (15.932 * math.pi ** 2):
            O = 15.932 * (math.pi ** 2) * I
        else:
            O = 1 / (15.932 * (math.pi ** 2) * I)

        # Measure Rectangularity
        minRect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(minRect)
        area = cv2.contourArea(contour)
        areaMinRect = cv2.contourArea(box)
        R = area / areaMinRect

        return [E, T, O, R]

    def extract_zernike(self, element, _radius):
        return mahotas.features.zernike_moments(resize(element, (64, 64)), radius=_radius, degree=10)


