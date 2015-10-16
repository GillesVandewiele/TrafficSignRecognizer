import os
from inout.fileparser import FileParser

from predict.benchmark import BenchmarkPredictor
from predict.prediction import Prediction

__author__ = 'Group16'

"""

    Main class of our project.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

pred = BenchmarkPredictor()
pred.train(os.path.join(os.path.dirname(__file__), "train"))
predictionObject = Prediction()
for testImage in os.listdir(os.path.join(os.path.dirname(__file__), "test")):
    predictionObject.addPrediction(pred.predict(os.path.join(os.path.dirname(__file__), "test", testImage)))

FileParser.write_CSV("benchmark.xlsx", predictionObject)
