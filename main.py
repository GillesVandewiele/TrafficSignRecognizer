import os
import numpy
from sklearn.cross_validation import KFold
from inout.fileparser import FileParser
from inout.transformations import Transformations

from predict.benchmark import BenchmarkPredictor
from predict.colorpredictor import ColorPredictor
from predict.prediction import Prediction

__author__ = 'Group16'

"""

    Main class of our project.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

"""
train_images_dir = os.path.join(os.path.dirname(__file__), "train")
d10_dir = os.path.join(train_images_dir, "blue_circles", "D10")
for image in os.listdir(d10_dir):
    img = FileParser.read_image(os.path.join(d10_dir, image))
    img_gray = img.convert("L")
    print("image = ", image, "; histogram = ")
    print(img.histogram())

    # Get indices of 10 greatest values of histogram
    n=10
    histogram = numpy.array(img.histogram())
    print(histogram.argsort()[-n:])

d5_dir = os.path.join(train_images_dir, "blue_circles", "D5")
for image in os.listdir(d5_dir):
    img = FileParser.read_image(os.path.join(d5_dir, image))
    img_gray = img.convert("L")
    print("image = ", image, "; histogram = ")
    print(img.histogram())

    # Get indices of 10 greatest values of histogram
    n=10
    histogram = numpy.array(img.histogram())
    print(histogram.argsort()[-n:])

number_images = sum([len(files) for r, d, files in os.walk(train_images_dir)])
kf = KFold(number_images, n_folds=2, shuffle=True)
for train, test in kf:
    print("%s %s", (train, test))


"""
def get_color_histogram_information(images_path, n):
    pass

def get_results(train_images_dir):
        results = []
        for shapesDirectory in os.listdir(train_images_dir):
            os.listdir(os.path.join(train_images_dir, shapesDirectory))
            for signDirectory in os.listdir(os.path.join(train_images_dir, shapesDirectory)):
                results.extend([signDirectory]*len(os.listdir(os.path.join(train_images_dir, shapesDirectory, signDirectory))))

        return results

# Directory of our training data (it's a mess in python...)
train_images_dir = os.path.join(os.path.dirname(__file__), "train")

# Get the results of the training data & a list of all images
results = get_results(train_images_dir)
train_images = []
for root, subFolders, files in os.walk(train_images_dir):
    for file in files:
        train_images.append(os.path.join(root, file))

# Decide on indices of training and validation data using k-fold cross validation
k = 2
number_images = len(train_images)
kf = KFold(number_images, n_folds=k, shuffle=True, random_state=1337)

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

# Color predictor
pred = ColorPredictor()
avg_logloss = 0
for train, validation in kf:
    train_set = [train_images[i] for i in train]
    validation_set = [train_images[i] for i in validation]
    train_set_results = [results[i] for i in train]
    validation_set_results = [results[i] for i in validation]

    prediction_object = Prediction()
    pred.train(train_set, train_set_results, 256)

    for elt in validation_set:
        prediction_object.addPrediction(pred.predict(elt))

    # Evaluate and add to logloss
    avg_logloss += prediction_object.evaluate(validation_set_results)


print("Average logloss score of the color predictor using ", k, " folds: ", avg_logloss/k)
"""
pred = BenchmarkPredictor()
pred.train()
predictionObject = Prediction()
for testImage in os.listdir(os.path.join(os.path.dirname(__file__), "test")):
    predictionObject.addPrediction(pred.predict(os.path.join(os.path.dirname(__file__), "test", testImage)))
"""

# Write out the output in the required format
#FileParser.write_CSV("benchmark.xlsx", predictionObject)
