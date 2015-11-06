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
    def __init__(self, _clusters, _image_size, _block_size):
        self.clusters = _clusters
        self.image_size = _image_size
        self.block_size = _block_size

    def extract_feature_vector(self, image):
        return self.calculateDCT(color.rgb2gray(image), self.clusters, self.image_size, self.block_size)

    def calculateDCT(self, img, clusters, image_size, block_size):

        # Read image in grayscale and convert to workable shape
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

        coefficients = []
        for i in range(int(image_size/block_size)):
            dct = fftpack.dct(fftpack.dct(newImg[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size].T, norm='ortho').T, norm='ortho')
            #coefficients.extend([dct[0, 0], dct[0, 1], dct[1, 0], dct[2, 0], dct[1, 1], dct[0, 2], dct[0, 3], dct[1, 2], dct[2, 1], dct[3, 0], dct[4, 0], dct[3, 1], dct[2, 2], dct[1, 3], dct[0, 4], dct[0, 5], dct[1, 4], dct[2, 3], dct[3, 2], dct[4, 1], dct[5, 0]])
            coefficients.extend(dct.reshape(-1))

        return coefficients / numpy.linalg.norm(coefficients)
