import os
from random import shuffle

import cv2
import lasagne
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
from numpy.ma import append
from skimage.transform import resize
from lasagne import layers
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from inout.fileparser import FileParser

from predict.hogfeatureextractor import HogFeatureExtractor
from predict.prediction import Prediction
from predict.siftfeatureextractor import SiftFeatureExtractor


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
 
 
def preprocess_image(image):
    image_array = cv2.imread(image)
    im = resize(image_array, (64, 64, 3))
    return im

def rotateImage(image, angle):
    return cv2.resize(image, (64, 64))
 
 
 
def build_mlp(nr_features):
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('hidden2', layers.DenseLayer),
            ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, nr_features),  # 96x96 input pixels per batch
        hidden_num_units=300,  # number of units in hidden layer
        hidden2_num_units=250,  # number of units in hidden layer
        hidden2_num_units=200,  # number of units in hidden layer
        output_nonlinearity=lasagne.nonlinearities.softmax,  # output layer uses identity function
        output_num_units=81,  # 30 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=100,  # we want to train this many epochs
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

    print("Loaded: ", len(train_images), " train images with ", len(train_results), " corresponding results and ",
          len(test_images), " with ", len(test_results), " corresponding results.")

    cfe = HogFeatureExtractor(8)
    sift_extractor = SiftFeatureExtractor()
    sift_extractor.set_codebook(train_images)

    feature_extractors = [cfe, sift_extractor]

    feature_vectors = []
    for image in train_images:
        print("Extracting features from training image ", image, "...")
        preprocessed_color_image = preprocess_image(image)
        feature_vector = []
        for feature_extractor in feature_extractors:
            if type(feature_extractor) != SiftFeatureExtractor:
                feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
            else:
                feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(image))
        feature_vectors.append(feature_vector)

    print("Feature reduction")
    print("From shape: ", len(feature_vectors), len(feature_vectors[0]))
    clf = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1)

    # Feature selection/reduction
    new_feature_vectors = clf.fit_transform(feature_vectors, train_results)
    print("To shape:", len(new_feature_vectors), len(new_feature_vectors[0]))
    X_train = np.asarray(new_feature_vectors)
    y_train = np.asarray(train_results)

    print("Building model")
    network = build_mlp(nr_features=len(new_feature_vectors[0]))
    print("Fitting")
    network.fit(X_train, y_train)

    print("Predicting")
    prediction_object = Prediction()
    for image in test_images:
        print("Extracting features from validating image ", image, "...")
        preprocessed_color_image = preprocess_image(image)
        validation_feature_vector = []
        for feature_extractor in feature_extractors:
            if type(feature_extractor) != SiftFeatureExtractor:
                validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
            else:
                validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(image))

        new_validation_feature_vector = clf.transform(validation_feature_vector)
        prediction_object.addPrediction(network.predict_proba(new_validation_feature_vector)[0])

    print("Logloss score = ", prediction_object.evaluate(test_results))

    """
    train_images = train_images[200:700]
    test_images = test_images[200:700]
    train_results = train_results[200:700]
    test_results = test_results[200:700]
    """

    """
    all_train_images = []
    for image in train_images:
        all_train_images.append(np.asarray(preprocess_image(image,cfe)))
    """
    """
    all_test_images = []
    for image in test_images:
        all_test_images.append(np.asarray(preprocess_image(image, cfe)))


    #X_test = np.asarray(all_test_images)
    #y_test = np.asarray(test_results)


    X_train = np.asarray(all_train_images)
    y_train = np.asarray(train_results)
    """
    """
    kf = KFold(len(train_images), n_folds=2, shuffle=True, random_state=13337)

    for train, validation in kf:
        # Divide the train_images in a training and validation set (using KFold)
        training_images = np.asarray([train_images[i%len(train_images)] for i in train])
        validating_images = np.asarray([train_images[i%len(train_images)] for i in validation])
        y_train = np.asarray([train_results[i%len(train_images)] for i in train])
        y_val = np.asarray([train_results[i%len(train_images)] for i in validation])

        cfe = HogFeatureExtractor(8)
        #sift_extractor = SiftFeatureExtractor()
        #sift_extractor.set_codebook(training_images)

        feature_extractors = [cfe]#, sift_extractor]
        feature_vectors = []
        for image in training_images:
            print("Extracting features from training image ", image, "...")
            preprocessed_color_image = preprocess_image(image)
            feature_vector = []
            for feature_extractor in feature_extractors:
                if type(feature_extractor) != SiftFeatureExtractor:
                    feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                else:
                    feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(image))
            feature_vectors.append(feature_vector)

        print("Feature reduction")
        print("From shape: ", len(feature_vectors), len(feature_vectors[0]))
        clf = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1)

        # Feature selection/reduction
        new_feature_vectors = clf.fit_transform(feature_vectors, y_train)
        print("To shape:", len(new_feature_vectors), len(new_feature_vectors[0]))
        X_train = np.asarray(new_feature_vectors)

        print("Building model")
        network = build_mlp(nr_features = len(new_feature_vectors[0]))
        print("Fitting")
        network.fit(X_train, y_train)

        print("Predicting")
        prediction_object = Prediction()
        for image in validating_images:
            print("Extracting features from validating image ", image, "...")
            preprocessed_color_image = preprocess_image(image)
            validation_feature_vector = []
            for feature_extractor in feature_extractors:
                if type(feature_extractor) != SiftFeatureExtractor:
                    validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                else:
                    validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(image))

            new_validation_feature_vector = clf.transform(validation_feature_vector)
            prediction_object.addPrediction(network.predict_proba(new_validation_feature_vector)[0])
        for prediction in range(len(predictions)):
            print("--------------------")
            print(prediction_object.TRAFFIC_SIGNS[y_val[prediction]], y_val[prediction])
            print("--------------------")
            prediction_object.addPrediction(predictions[prediction])

        #print("Logloss score = ", prediction_object.evaluate(y_val))
        """

main()