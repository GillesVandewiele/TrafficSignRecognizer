import os

from predict.predictor import Predictor
from predict.prediction import Prediction

__author__ = 'Group16'

"""

    This class contains the benchmark predictor. For the training data, calculate the occurrence of each traffic
    sign and assign these occurrence probabilities to the new test data

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

class BenchmarkPredictor(Predictor):

    def __init__(self):
        self.occurrenceProbabilities = {}

    def train(self, trainingData):
        counter = 0

        for shapesDirectory in os.listdir(trainingData):
            os.listdir(os.path.join(trainingData, shapesDirectory))
            for signDirectory in os.listdir(os.path.join(trainingData, shapesDirectory)):
                self.occurrenceProbabilities[signDirectory] = len(os.listdir(os.path.join(trainingData, shapesDirectory, signDirectory)))
                counter += len(os.listdir(os.path.join(trainingData, shapesDirectory, signDirectory)))


        for occurrenceProb in self.occurrenceProbabilities:
            self.occurrenceProbabilities[occurrenceProb] = self.occurrenceProbabilities[occurrenceProb]/counter

    def predict(self, image):
        return self.occurrenceProbabilities

pred = BenchmarkPredictor()
pred.train(os.path.join(os.path.dirname(__file__), "../train"))
