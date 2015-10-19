__author__ = 'Group16'

"""

    Interface for the Predictor class

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

class Predictor(object):

    def __init__(self):
        pass

    def train(self, trainingData, results):
        """
        Train the data.
        :param trainingData: The filenames. Should have same length as results and correspond.
        :param results: The results (string such as D10, D1e, ...). Should have same length as results and correspond.
        """
        pass

    def predict(self, image):
        pass