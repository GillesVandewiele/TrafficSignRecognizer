import os
import numpy
import skimage
from sklearn.cross_validation import KFold
from inout.fileparser import FileParser
from inout.transformations import Transformations
from skimage.io import imread

from predict.benchmark import BenchmarkPredictor
from predict.colorpredictor import ColorPredictor
from predict.prediction import Prediction

__author__ = 'Group16'

"""

    Main class of our project.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""
def get_results(train_images_dir):
        results = []
        for shapesDirectory in os.listdir(train_images_dir):
            os.listdir(os.path.join(train_images_dir, shapesDirectory))
            for signDirectory in os.listdir(os.path.join(train_images_dir, shapesDirectory)):
                results.extend([signDirectory]*len(os.listdir(os.path.join(train_images_dir, shapesDirectory, signDirectory))))

        return results

# Directory of our training data (it's a mess in python...)
train_images_dir = os.path.join(os.path.dirname(__file__), "train")
test_images_dir = os.path.join(os.path.dirname(__file__), "test")

# Get the results of the training data & a list of all images
results = get_results(train_images_dir)
train_images = []
for root, subFolders, files in os.walk(train_images_dir):
    for file in files:
        train_images.append(os.path.join(root, file))

test_images = []
for root, subFolders, files in os.walk(test_images_dir):
    for file in files:
        test_images.append(os.path.join(root, file))

# Decide on indices of training and validation data using k-fold cross validation
k = 2
number_images = len(train_images)
kf = KFold(100, n_folds=k, shuffle=True, random_state=1337)

"""
# Benchmark predictor
pred = BenchmarkPredictor()
avg_logloss = 0
for train, validation in kf:
    train_set = [train_images[i] for i in train]
    validation_set = [train_images[i] for i in validation]
    train_set_results = [results[i] for i in train]
    validation_set_results = [results[i] for i in validation]

    prediction_object = Prediction()
    pred.train(train_set, train_set_results)
    for elt in validation_set:
        prediction_object.addPrediction(pred.predict(elt), True)

    # Evaluate and add to logloss
    avg_logloss += prediction_object.evaluate(validation_set_results)

print("Average logloss score of the benchmark predictor using ", k, " folds: ", avg_logloss/k)
"""

train_images = train_images[0:100]
results = results[0:100]

# Color predictor
pred = ColorPredictor()
avg_logloss = 0
for train, validation in kf:
    train_set = [train_images[i] for i in train]
    validation_set = [train_images[i] for i in validation]
    train_set_results = [results[i] for i in train]
    validation_set_results = [results[i] for i in validation]

    prediction_object = Prediction()

    pred.train(train_set, train_set_results)

    print(pred.histograms)

    for elt in validation_set:
        prediction_object.addPrediction(pred.predict(elt))

    # Evaluate and add to logloss
    avg_logloss += prediction_object.evaluate(validation_set_results)

print("Average logloss score of the color predictor using ", k, " folds: ", avg_logloss/k)

"""
pred = ColorPredictor()
pred.extract_hue_histogram(os.path.join(os.path.dirname(__file__), "test.png"))
pred.extract_hue_histogram(os.path.join(os.path.dirname(__file__), "00917_11555.png"))
"""


# Run it on the test set and write out the output in the required format
"""
pred = ColorPredictor()

pred.train(train_images, results)

prediction_object = Prediction()
for elt in test_images:
    prediction_object.addPrediction(pred.predict(elt))

FileParser.write_CSV("color_chances.xlsx", prediction_object)
prediction_object.adapt_probabilities()
FileParser.write_CSV("color_nochances.xlsx", prediction_object)
"""
