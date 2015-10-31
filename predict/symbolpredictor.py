import numpy
import operator
from skimage.io import imread
from sklearn import cluster
from scipy.misc import imresize
from scipy import fftpack
from predict.predictor import Predictor
__author__ = 'Group 16'

"""

    This class contains a predictor that uses feature extraction.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

class SymbolPredictor(Predictor):
    def __init__(self):
        self.dcts = []

    def train(self, trainingData, results):
        """
        Train the data.
        :param trainingData: The filenames. Should have same length as results and correspond.
        :param results: The results (string such as D10, D1e, ...). Should have same length as results and correspond.
        """

        self.dcts = []

        for element in range(len(trainingData)):
            print("Training ", trainingData[element], "...")

            self.dcts.append([self.calculateDCT(trainingData[element]), results[element]])

    def calculateDCT(self, element):
        clusters = 3

        # Read image in grayscale and convert to workable shape
        img = imread(element, True)
        imgArr = img.reshape((-1, 1))

        # Apply K-means to image
        k_means = cluster.KMeans(n_clusters=clusters)
        k_means.fit(imgArr)
        values = k_means.cluster_centers_.squeeze()

        # Change colors to equidistant shades of gray
        values_ = sorted(values)
        for j in range(clusters):
            index = numpy.where(values == values_[j])
            values[index] = j * 1 / (clusters - 1)

        # Create new image
        newImg = numpy.choose(k_means.labels_, values)
        newImg.shape = img.shape

        # Resize image for DCT
        newImg = imresize(newImg, (128, 128)).astype(float)

        # DCT of image
        dct = fftpack.dct(fftpack.dct(newImg.T, norm='ortho').T, norm='ortho')

        # Return 15 most dominant coefficients
        return [dct[0, 0], dct[0, 1], dct[1, 0], dct[2, 0], dct[1, 1], dct[0, 2], dct[0, 3], dct[1, 2], dct[2, 1], dct[3, 0], dct[4, 0], dct[3, 1], dct[2, 2], dct[1, 3], dct[0, 4]]

"""
    def predict(self, image):

        print("Predicting ", image, "...")

        dct = self.calculateDCT(image)

        # Now calculate the MSE to all other element
        mse_values = []
        for i in range(len(self.dcts)):
            mse_values[i] = sum([(dct[x]-self.dcts[i, 0, x]) ** 2 for x in range(len(dct))])

        sortedNeighbours = sorted(self.dcts[:][1], key=mse_values)

        k_neighbours = 5

        probabilities = {}
        for i in range(k_neighbours):
            if sortedNeighbours[i] in probabilities:
                probabilities[sortedNeighbours[i]] += 1 / k_neighbours
            else:
                probabilities[sortedNeighbours[i]] = 1 / k_neighbours

        print(sorted(probabilities.items(), key=operator.itemgetter(1), reverse=True))
        return probabilities
"""