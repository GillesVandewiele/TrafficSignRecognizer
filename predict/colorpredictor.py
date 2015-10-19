import operator
import math
from numpy import histogram
from skimage.io import imread
from predict.predictor import Predictor
__author__ = 'Group 16'

"""

    This class contains a predictor that uses colour information.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

class ColorPredictor(Predictor):

    def __init__(self):
        self.histograms = {}

    def train(self, trainingData, results, nBins):
        """
        Train the data.
        :param trainingData: The filenames. Should have same length as results and correspond.
        :param results: The results (string such as D10, D1e, ...). Should have same length as results and correspond.
        """
        histogram_sums = {}
        histogram_counters = {}
        counter = 0
        for element in trainingData:
            img = imread(element)
            histogram_temp = histogram(img, bins=nBins)
            if results[counter] not in histogram_sums:
                histogram_sums[results[counter]] = histogram_temp
                histogram_counters[results[counter]] = 1
            else:
                histogram_sums[results[counter]] = [x + y for x, y in zip(histogram_sums[results[counter]], histogram_temp)]
                histogram_counters[results[counter]] += 1
            counter += 1

        for element in histogram_sums:
            self.histograms[element] = [x / histogram_counters[element] for x in histogram_sums[element]]

    #TODO: calculate histogram for each color channel seperately!
    def predict(self, image):
        nBins = len(list(self.histograms.values())[0][0])
        img = imread(image)
        test_image_histogram = histogram(img, bins=nBins)
        mse_values = {}
        mse_sum = 0

        print("image = ", image)

        for element in self.histograms:
            mse_value = 0
            for bin in range(nBins):
                mse_value += pow(self.histograms[element][0][bin] - test_image_histogram[0][bin], 2)
            mse_sum += mse_value
            if element not in mse_values:
                mse_values[element] = mse_value
            else:
                mse_values[element] += mse_value

        mse_log_sum = 0
        for value in mse_values:
            mse_values[value] = math.log(mse_values[value]/mse_sum)*-1
            mse_log_sum += mse_values[value]

        norm_mse_values = {}
        for value in mse_values:
             norm_mse_values[value] = mse_values[value]/mse_log_sum

        print(sorted(norm_mse_values.items(), key=operator.itemgetter(1), reverse=True))
        print(sorted(mse_values.items(), key=operator.itemgetter(1), reverse=True))

        return norm_mse_values
