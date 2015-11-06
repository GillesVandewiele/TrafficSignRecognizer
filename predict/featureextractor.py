__author__ = 'Group16'

"""

    Interface for the Feature Extractor class

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

class FeatureExtractor(object):

    def __init__(self):
        raise NotImplementedError("A constructor must be implemented!")

    def extract_feature_vector(self, image):
        raise NotImplementedError("The extract feature vector function must be implemented")