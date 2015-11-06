import numpy
from skimage import color
from sklearn import cluster
from scipy.misc import imresize
from scipy import fftpack
from predict.featureextractor import FeatureExtractor
__author__ = 'Group 16'

"""
    This class contains the code to extract DC coefficients from the images after applying Discrete Cosine Transform
    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning
"""

class SymbolFeatureExtractor(FeatureExtractor):
    def __init__(self, _clusters, _image_size, _n_coeff):
        self.clusters = _clusters
        self.image_size = _image_size
        self.n_coeff = _n_coeff

    def extract_feature_vector(self, image):
        return self.calculateDCT(color.rgb2gray(image), self.clusters, self.image_size, self.n_coeff)

    def calculateDCT(self, img, clusters, image_size, n_coeff):

        # Convert image to workable shape
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
        newImg = newImg.astype(float)

        dct = fftpack.dct(fftpack.dct(newImg.T, norm='ortho').T, norm='ortho')

        # Order the coefficients according to their energy by zigzag traversing the matrix
        coefficients = [numpy.fliplr(dct).diagonal(i).tolist() for i in range(image_size - 1, -image_size, -1)]
        coefficients[0:len(coefficients):2] = [numpy.flipud(coefficients[i]).tolist() for i in range(0, len(coefficients), 2)]
        coefficients = [coefficients[i][j] for i in range(len(coefficients)) for j in range(len(coefficients[i]))]

        if n_coeff > 0:
            coefficients = coefficients[0:n_coeff]

        return coefficients / numpy.linalg.norm(coefficients)