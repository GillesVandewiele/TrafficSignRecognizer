import colorsys
import operator
import math
import numpy
from skimage.feature import hog
from skimage import color, exposure
from numpy import histogram, asarray, pad
from skimage.io import imread, imsave
from predict.predictor import Predictor
from skimage.transform import resize
__author__ = 'Group 16'

"""

    This class contains a predictor that uses colour information.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

class ColorFeatureExtractor(Predictor):

    def extract_hog(self, element):
        image = resize(color.rgb2gray(imread(element)), (64, 64))
        fd = hog(image, orientations=9, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), normalise=True)
        return fd.tolist()

    def calculate_histogram(self, hue, bins=20):
        hist = histogram(hue, bins=bins, range=(0, 1))

        # DEBUG: Save our results
        ##imsave(element[:-4]+'test.png', asarray(hue))

        # Red 250, Yellow 35, Blue 150-160
        return [x/sum(hist[0]) for x in hist[0]]

    def extract_hue(self, element, binary=False, debug=False):

        # Read image as array with RGB values
        img = imread(element)

        # Converting the RGB values to HSV values
        hsv = [None]*len(img)
        for i in range(len(img)):
            temp = [None]*len(img[0])
            for j in range(len(img[0])):
                temp[j] = colorsys.rgb_to_hsv(img[i, j, 0], img[i, j ,1], img[i, j, 2])
            hsv[i] = temp

        # Extracting the 3 channels
        hue = [None]*len(img)
        saturation = [None]*len(img)
        value = [None]*len(img)
        for x in range(len(img)):
            hue_temp = [None]*len(img[0])
            saturation_temp = [None]*len(img[0])
            value_temp = [None]*len(img[0])
            for y in range(len(img[0])):
                hue_temp[y] = hsv[x][y][0]
                saturation_temp[y] = hsv[x][y][1]
                value_temp[y] = hsv[x][y][2]
            hue[x] = hue_temp
            saturation[x] = saturation_temp
            value[x] = value_temp

        # Normalize the Value values
        value_norm = [None]*len(value)
        for x in range(len(value)):
            value_norm_temp = [None]*len(value[x])
            for y in range(len(value[x])):
                value_norm_temp[y] = value[x][y]/255
            value_norm[x] = value_norm_temp
        value = value_norm

        # We now loop through the hue values and filter out the ones with red values and set them to white
        # To avoid inconsistencies, we need to make sure the hue value lays in his chromatic area,
        # else we set the pixel to black
        for x in range(len(hue)):
            for y in range(len(hue[x])):
                if (hue[x][y]>0.95 and hue[x][y]<=1) or (hue[x][y]>=0 and hue[x][y]<0.05):
                    hue[x][y] = 1
                elif binary:
                    hue[x][y]=0

                if saturation[x][y] < 0.25:  # Achromatic area
                    hue[x][y] = 0
                if value[x][y] < 0.2 or value[x][y] > 0.9:  # Achromatic area
                    hue[x][y] = 0

        if debug:
            imsave(element[:-4]+'test.png', asarray(hue))

        return hue
