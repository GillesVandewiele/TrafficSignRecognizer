from math import sqrt
import cv2
from numpy import zeros_like, vstack, resize, zeros, histogram
from scipy.cluster import vq
from predict.featureextractor import FeatureExtractor
from vlfeat_wrapper import process_image, read_features_from_file

__author__ = 'Group 16'

"""

    This class contains an extractor that extracts colour information. It converts a hue image from a color image and
    calculate a histogram of these values.

    Thanks to:
        https://github.com/shackenberg/Minimal-Bag-of-Visual-Words-Image-Classifier

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""


class SiftFeatureExtractor(FeatureExtractor):

    def __init__(self):
        self.PRE_ALLOCATION_BUFFER = 1000  # for sift
        self.K_THRESH = 1.0

    def set_codebook(self, train_images):
        self.codebook = self.get_code_book(train_images)

    def extract_feature_vector(self, image):
        return self.computeHistograms(self.codebook, self.dict2numpy(self.extractSift([image])))

    def computeHistograms(self, codebook, descriptors):
        code, dist = vq.vq(descriptors, codebook)
        histogram_of_words, bin_edges = histogram(code,
                                                  bins=range(codebook.shape[0] + 1),
                                                  normed=True)
        return histogram_of_words

    def extractSift(self, input_files):
        all_features_dict = {}
        for i, fname in enumerate(input_files):
            print("calculating sift features for", fname)
            if type(fname) == str:
                image_array = cv2.imread(fname)
                image_array = resize(image_array, (64, 64, 3))
            else:
                image_array = fname;
            process_image(image_array, 'tmp.sift')
            locs, descriptors = read_features_from_file('tmp.sift')
            all_features_dict[fname] = descriptors
        return all_features_dict

    def dict2numpy(self, dict):
        nkeys = len(dict)
        array = zeros((nkeys * self.PRE_ALLOCATION_BUFFER, 128))
        pivot = 0
        for key in dict.keys():
            value = dict[key]
            nelements = value.shape[0]
            while pivot + nelements > array.shape[0]:
                padding = zeros_like(array)
                array = vstack((array, padding))
            array[pivot:pivot + nelements] = value
            pivot += nelements
        array = resize(array, (pivot, 128))
        return array

    def get_code_book(self, train_images):
        features = self.extractSift(train_images)
        all_features_array = self.dict2numpy(features)
        nfeatures = all_features_array.shape[0]
        nclusters = int(sqrt(nfeatures))
        codebook, distortion = vq.kmeans(all_features_array,
                                                 nclusters,
                                                 thresh=self.K_THRESH)
        return codebook