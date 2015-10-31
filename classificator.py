import os
# import cv2
from numpy import asarray, pad, resize, append
from skimage.io import imread
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from inout.fileparser import FileParser
from predict.colorpredictor import ColorPredictor
from predict.prediction import Prediction
#from predict.shapepredictor import ShapePredictor
from predict.symbolpredictor import SymbolPredictor

__author__ = 'Group16'

"""

    Main class of our project.

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

def get_results(train_images_dir):
        # Check all files in the directory, the parent directory of a photo is the label
        results = []
        for shapesDirectory in os.listdir(train_images_dir):
            os.listdir(os.path.join(train_images_dir, shapesDirectory))
            for signDirectory in os.listdir(os.path.join(train_images_dir, shapesDirectory)):
                results.extend([signDirectory]*len(os.listdir(os.path.join(train_images_dir, shapesDirectory, signDirectory))))

        return results

def get_images_from_directory(directory):
    # Get all files from a directory
    images = []
    for root, subFolders, files in os.walk(directory):
        for file in files:
            images.append(os.path.join(root, file))

    return images

def transform_classes(results):
    # Reduce the number of classes in the results for color classification, the resulting classes are:
    #   red signs (red_circles, stop, forbidden, others: F41)
    #   semi-red signs (triangles, red_blue_circles, reversed triangles, others: F43)
    #   blue signs (blue_circles, rectangles_down, rectangles_up, squares, others: F13, F21, F23A, F25, F27, F29, Handic)
    #   white signs (other: C37, F1, F1a_h, F33_34, F3a_h, F4b, begin, end, e0c, lang, m)
    #   others (diamonds, others: F31, F35)
    new_classes = []
    for result in results:
        if result in ["B19", "C3", "C11", "C21", "C23", "C29", "C31", "C35", "C43", "F4a", "F41", "C1", "B5",
                      "B1", "B3", "B7", "E1", "E5", "E3", "E7", "A1AB", "A1CD", "A7A", "A7B", "A13", "A14", "A15",
                      "A23", "A25", "A29", "A31", "A51", "B15A", "B17", "F43"]:
            new_classes.append("red")
        #elif result in ["B1", "B3", "B7", "E1", "E5", "E3", "E7", "A1AB", "A1CD", "A7A", "A7B", "A13", "A14", "A15",
        #                "A23", "A25", "A29", "A31", "A51", "B15A", "B17", "F43"]:
        #    new_classes.append("semi-red")
        #elif result in ["D1a", "D1b", "D1e", "D5", "D7", "D9", "D10", "F12a", "F12b", "B21", "E9a", "E9a_miva", "E9b",
        #                "E9cd", "E9e", "F45", "F47", "F59", "X", "F19", "F49", "F50", "F87", "F13", "F21", "F23A",
        #                "F25", "F27", "F29", "Handic"]:
        #    new_classes.append("blue")
        #elif result in ["C37", "F1", "F1a_h", "F33_34", "F3a_h", "F4b", "begin", "end", "e0c", "lang", "m"]:
        #    new_classes.append("white")
        else:
            new_classes.append("others")
    return new_classes

def classify_traffic_signs(k, excel_path):
    # Get the images and results from the directories train and test
    train_images_dir = os.path.join(os.path.dirname(__file__), "train")
    test_images_dir = os.path.join(os.path.dirname(__file__), "test")

    train_images = get_images_from_directory(train_images_dir)
    results = get_results(train_images_dir)

    test_images = get_images_from_directory(test_images_dir)

    # Decide on indices of training and validation data using k-fold cross validation
    """len(train_images)"""
    #kf = KFold(len(train_images), n_folds=k, shuffle=True, random_state=1337)
    kf = KFold(300, n_folds=k, shuffle=True, random_state=1337)
    train_images=train_images[600:900]
    results=results[600:900]

    # Predict
    avg_logloss = 0
    for train, validation in kf:
        # Divide the train_images in a training and validation set (using KFold)
        train_set = [train_images[i] for i in train]
        validation_set = [train_images[i] for i in validation]
        train_set_results = [results[i] for i in train]
        validation_set_results = [results[i] for i in validation]
        # Iterate over the training set and transform each input vector to a feature vector
        feature_vectors = []
        color_extractor = ColorPredictor()
        #shape_extractor = ShapePredictor()
        symbol_extractor = SymbolPredictor()
        #print(transform_classes(train_set_results))
        for image in train_set:

            print("Training ", image, "...")

            # First we extract all features that got smth to do with color
            #hue = color_extractor.extract_hue(image)
            #hue_binary = color_extractor.extract_hue(image, binary=True)
            #feature_vector = color_extractor.calculate_histogram(hue_binary, 2)
            #hue = resize(hue, (32, 32))
            #mom = cv2.moments(asarray(hue))
            #print(mom)
            #hu = cv2.HuMoments(mom)
            #print(hu)
            #feature_vector = hu.flatten()
            #print(hu)

            #TODO: extract shape features and extend the feature_vector
            #shape_features = shape_extractor.predictShape(hue)
            #feature_vector = append(feature_vector, shape_features)

            #TODO: extract symbol/icon features
            feature_vector = symbol_extractor.calculateDCT(image)

            feature_vectors.append(feature_vector)

        # We use C-SVM with a linear kernel and want to predict probabilities
        # max_iter = -1 for no limit on iterations (tol is our stopping criterion)
        # Put verbose off for some output and don't use the shrinking heuristic (needs some testing)
        # Allocate 1 GB of memory for our kernel
        # We are using seed 1337 to always get the same results (can be put on None for testing)
        clf = SVC(C=1.0, cache_size=3000, class_weight=None, kernel='linear', max_iter=-1, probability=True,
          random_state=1337, shrinking=False, tol=0.001, verbose=False)
        #clf.fit(feature_vectors, transform_classes(train_set_results))
        clf.fit(feature_vectors, train_set_results)

        prediction_object = Prediction()
        print("test")
        for im in validation_set:
            print("Predicting ", im, "...")
            #hue = color_extractor.extract_hue(im)
            #hue_binary = color_extractor.extract_hue(im, binary=True)
            #validation_feature_vector = color_extractor.calculate_histogram(hue_binary, 2)
            #mom = cv2.moments(asarray(hue).flatten())
            #hu = cv2.HuMoments(mom)
            #validation_feature_vector = hu
            #validation_feature_vector.append(shape_extractor.predictShape(hue))
            validation_feature_vector = symbol_extractor.calculateDCT(im)
            print(clf.predict_proba(validation_feature_vector)[0])
            prediction_object.addPrediction(clf.predict_proba(validation_feature_vector)[0])


        # Evaluate and add to logloss
        #print(prediction_object.evaluate(transform_classes(validation_set_results)))
        #avg_logloss += prediction_object.evaluate(transform_classes(validation_set_results))
        print(prediction_object.evaluate(validation_set_results))
        avg_logloss += prediction_object.evaluate(validation_set_results)

    print("Average logloss score of the predictor using ", k, " folds: ", avg_logloss/k)
    FileParser.write_CSV(excel_path, prediction_object)
"""
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
frame = imread(os.path.join(os.path.dirname(__file__), "00129_02203.png"))
cv2.imshow('frame', frame)
input("")
fgmask = fgbg.apply(frame)
cv2.imshow('frame',fgmask)
input("Press Enter to continue...")
"""
classify_traffic_signs(2, "testing.xlsx")