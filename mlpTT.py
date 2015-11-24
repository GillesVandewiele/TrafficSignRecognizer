import os
from random import shuffle

import cv2
import lasagne
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
from skimage.transform import resize
from lasagne import layers

from predict.hogfeatureextractor import HogFeatureExtractor
from predict.prediction import Prediction


def get_results(train_images_dir):
        # Check all files in the directory, the parent directory of a photo is the label
        results = []
        pred = Prediction()
        for shapesDirectory in os.listdir(train_images_dir):
            os.listdir(os.path.join(train_images_dir, shapesDirectory))
            for signDirectory in os.listdir(os.path.join(train_images_dir, shapesDirectory)):
                dir = [signDirectory]*len(os.listdir(os.path.join(train_images_dir, shapesDirectory, signDirectory)))
                for el in dir:
                    results.append(pred.TRAFFIC_SIGNS.index(el))
 
        return results
 
def get_images_from_directory(directory):
    # Get all files from a directory
    images = []
    for root, subFolders, files in os.walk(directory):
        for file in files:
            images.append(os.path.join(root, file))
 
    return images
 
 
def preprocess_image(image, extractor):
    image_array = cv2.imread(image)
    im = resize(image_array, (64, 64, 3))
    return extractor.extract_feature_vector(im)
 
 
 
def build_mlp():
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 576),  # 96x96 input pixels per batch
        hidden_num_units=100,  # number of units in hidden layer
        output_nonlinearity=lasagne.nonlinearities.softmax,  # output layer uses identity function
        output_num_units=81,  # 30 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=40,  # we want to train this many epochs
        verbose=1,
    )
    return net1

def main(num_epochs=100):
    # Load the dataset
    print("Loading data...")
 
    train_images_dir = os.path.join(os.path.dirname(__file__), "train")
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_labeled")
    train_images = get_images_from_directory(train_images_dir)
    test_images = get_images_from_directory(test_images_dir)
    train_results = get_results(train_images_dir)
    test_results = get_results(test_images_dir)
    cfe = HogFeatureExtractor(8)
   
    all_train_images = []
    for image in train_images:
        all_train_images.append(np.asarray(preprocess_image(image,cfe)))

    all_test_images = []
    for image in test_images:
        all_test_images.append(np.asarray(preprocess_image(image, cfe)))

    X_test = np.asarray(all_test_images)
    y_test = np.asarray(test_results)


    X_train = np.asarray(all_train_images)
    y_train = np.asarray(train_results)

    # Shuffle the data (because batches are used under the hood of NeuralNet) #TODO: do this??
    X_train_shuf = []
    y_train_shuf = []
    print(len(X_train))
    index_shuf = list(range(len(X_train)))
    shuffle(index_shuf)
    for i in index_shuf:
        X_train_shuf.append(X_train[i])
        y_train_shuf.append(y_train[i])

    X_train = np.asarray(X_train_shuf)
    y_train = np.asarray(y_train_shuf)

    # Create neural network model (depending on first command line parameter)
    print("Building model")
    network = build_mlp()

    print("Fitting")
    network.fit(X_train, y_train)

    print("Predicting")
    predictions = network.predict_proba(X_test)
    prediction_object = Prediction()
    print(predictions)
    print(predictions.shape)

    print("Logloss score = ", prediction_object.evaluate(test_results))

main()