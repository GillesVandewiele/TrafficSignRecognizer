import colorsys
import operator
import math
import numpy
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

class ColorPredictor(Predictor):
    """
    :var histograms: Dictionary with keys the name of sign and as value an array of size 3 with histograms for R, G & B
    """

    def __init__(self):
        self.histograms = {}

    def train(self, trainingData, results):
        """
        Train the data.
        :param trainingData: The filenames. Should have same length as results and correspond.
        :param results: The results (string such as D10, D1e, ...). Should have same length as results and correspond.
        """


        # TODO: find a good way to extract information out of the histograms of the training data!!
        self.histograms = {}
        counters = {}
        for element in range(len(trainingData)):
            print("Training ", trainingData[element], "...")

            if results[element] not in self.histograms:
                self.histograms[results[element]] = self.extract_hue(trainingData[element])[0]
                counters[results[element]] = 1
            else:
                self.histograms[results[element]] = [ x + y for x, y in zip(self.histograms[results[element]],
                                                                            self.extract_hue(trainingData[element])[0]) ]
                counters[results[element]] += 1

        for element in self.histograms:
            self.histograms[element] = [x/counters[element] for x in self.histograms[element]]

    def calculate_histogram(self, hue, bins=20):
        hist = histogram(hue, bins=bins, range=(0, 1))

        # DEBUG: Save our results
        ##imsave(element[:-4]+'test.png', asarray(hue))

        # Red 250, Yellow 35, Blue 150-160
        return [x/sum(hist[0]) for x in hist[0]]

    def extract_hue(self, element):

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
                else:
                    hue[x][y]=0
                if saturation[x][y] < 0.25:  # Achromatic area
                    hue[x][y] = 0
                if value[x][y] < 0.2 or value[x][y] > 0.9:  # Achromatic area
                    hue[x][y] = 0

        # TODO: apply region growing algorithm for even better performance!!

        ##imsave(element[:-4]+'test.png', asarray(hue))

        return hue


    def getChanceOnColor(self, color, histogram):
        # Histogram should contain 20 bins equally spaced from 0 to 1 (0, 0.05, 0.1, ... )
        if color == "red":
            return histogram[19]
        if color == "blue":
            return histogram[12] + histogram[13]
        if color == "yellow":
            return histogram[3]
        if color == "black":
            return histogram[0]
        if color == "white":
            return histogram[0]

    def predict(self, image):

        print("Predicting ", image, "...")

        hist = self.extract_hue(image)

        # Extract histogram
        hist = [x/sum(hist[0]) for x in hist[0]]

        # Now calculate the MSE to all other element
        mse_values = {}
        for element in self.histograms:
            mse_values[element] = sum([abs(hist[x]-self.histograms[element][x]) for x in range(len(hist))])

        # Reverse normalise the values
        for element in self.histograms:
            mse_values[element] /= sum(mse_values.values())
            mse_values[element] = -math.log(mse_values[element])

        #TODO: probabilities are too close to eachother, apply penalties!

        probabilities = {}
        for element in mse_values:
            probabilities[element] = mse_values[element] / sum(mse_values.values())

        print(sorted(probabilities.items(), key=operator.itemgetter(1), reverse=True))
        return probabilities
