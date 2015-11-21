import os
import cv2
from numpy import append
from skimage.transform import resize
from skimage import color
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from inout.fileparser import FileParser
from predict.colorfeatureextractor import ColorFeatureExtractor
from predict.prediction import Prediction
from predict.shapefeatureextractor import ShapeFeatureExtractor
from predict.siftfeatureextractor import SiftFeatureExtractor
from predict.symbolfeatureextractor import SymbolFeatureExtractor

__author__ = 'Group16'

"""
    The main class of our project. Contains code to perform k-fold cross validation for local testing and the code
    to make a submission on Kaggle.
    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning
"""

class TrafficSignRecognizer(object):

    def __init__(self):
        pass

    def get_results(self, train_images_dir):
            # Check all files in the directory, the parent directory of a photo is the label
            results = []
            for shapesDirectory in os.listdir(train_images_dir):
                os.listdir(os.path.join(train_images_dir, shapesDirectory))
                for signDirectory in os.listdir(os.path.join(train_images_dir, shapesDirectory)):
                    results.extend([signDirectory]*len(os.listdir(os.path.join(train_images_dir, shapesDirectory, signDirectory))))

            return results

    def get_images_from_directory(self, directory):
        # Get all files from a directory
        images = []
        for root, subFolders, files in os.walk(directory):
            for file in files:
                images.append(os.path.join(root, file))

        return images

    def preprocess_image(self, image, size):
        # Preprocess our image, creating a denoised color and gray image resized to a square size "size"
        image_array = cv2.imread(image)
        #denoised_image = cv2.fastNlMeansDenoisingColored(image_array,None,7,7,11,21)
        image_array = resize(image_array, (size, size, 3))
        return image_array

    def make_submission(self, train_images_path, test_images_path, output_file_path, feature_extractors, size=64):
        # Extract the train images with corresponding results & the test images
        train_images = self.get_images_from_directory(train_images_path)
        results = self.get_results(train_images_path)
        test_images = self.get_images_from_directory(test_images_path)

        # Create a vector of feature vectors (a feature matrix)
        feature_vectors = []
        for image in train_images:
            print("Training ", image, "...")
            preprocessed_color_image = self.preprocess_image(image, size)
            feature_vector = []
            for feature_extractor in feature_extractors:
                feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
            feature_vectors.append(feature_vector)

        # Using logistic regression as linear model to fit our feature_vectors to our results
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=32, intercept_scaling=1, solver='liblinear', max_iter=100,
                         multi_class='ovr', verbose=0)

        #clf = RandomForestClassifier(n_estimators=2500)

        # Logistic Regression for feature selection, higher C = more features will be deleted
        clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4)

        # Feature selection/reduction
        new_feature_vectors = clf2.fit_transform(feature_vectors, results)

        # Model fitting
        clf.fit(new_feature_vectors, results)

        # Iterate over the test images and add their prediction to a prediction object
        prediction_object = Prediction()
        for im in test_images:
            print("Predicting ", im)
            preprocessed_color_image = self.preprocess_image(im, size)
            validation_feature_vector = []
            for feature_extractor in feature_extractors:
                validation_feature_vector = append(validation_feature_vector,
                                                   feature_extractor.extract_feature_vector(preprocessed_color_image))
            new_validation_feature_vector = clf2.transform(validation_feature_vector)
            prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])

        # Write out the prediction object
        FileParser.write_CSV(output_file_path, prediction_object)


    def local_test(self, train_images_path, feature_extractors, k=2, nr_data_augments=1, size=64):
        # Extract the train images with corresponding results
        train_images = self.get_images_from_directory(train_images_path)
        results = self.get_results(train_images_path)

        kf = KFold(len(train_images)*nr_data_augments, n_folds=k, shuffle=True, random_state=1337)
        #kf = KFold(500, n_folds=k, shuffle=True, random_state=1337)
        train_errors = []
        test_errors = []

        for train, validation in kf:
            # Divide the train_images in a training and validation set (using KFold)
            train_set = [train_images[i%len(train_images)] for i in train]
            validation_set = [train_images[i%len(train_images)] for i in validation]
            train_set_results = [results[i%len(train_images)] for i in train]
            validation_set_results = [results[i%len(train_images)] for i in validation]

            # Create a vector of feature vectors (a feature matrix)
            feature_vectors = []
            counter=1
            for image in train_set:
                print("Training image ", image)
                counter += 1
                preprocessed_color_image = self.preprocess_image(image, size)
                feature_vector = []
                for feature_extractor in feature_extractors:
                        feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(image))
                feature_vectors.append(feature_vector)

            # Using logistic regression as linear model to fit our feature_vectors to our results
            clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=32, intercept_scaling=1, solver='liblinear', max_iter=100,
                             multi_class='ovr', verbose=0)

            #clf = RandomForestClassifier(n_estimators=100)

            # Logistic Regression for feature selection, higher C = more features will be deleted
            clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4)
            """
            reduction = Lasso()
            scaler = StandardScaler()
            X = scaler.fit_transform(feature_vectors)
            le = preprocessing.LabelEncoder()
            le.fit(train_set_results)
            reduction.fit(X, le.transform(train_set_results))
            print(reduction.coef_)
            """

            # Feature selection/reduction
            new_feature_vectors = clf2.fit_transform(feature_vectors, train_set_results)

            #clf.coef_ = reduction.coef_
            # Model fitting
            clf.fit(new_feature_vectors, train_set_results)

            train_prediction_object = Prediction()
            counter=0
            for im in train_set:
                print("predicting train image ", counter)
                counter+=1
                preprocessed_color_image = self.preprocess_image(im, size)
                validation_feature_vector = []
                for feature_extractor in feature_extractors:
                    if type(feature_extractor) != SiftFeatureExtractor:
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                    else:
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(image))
                new_validation_feature_vector = clf2.transform(validation_feature_vector)
                train_prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])

            test_prediction_object = Prediction()
            counter=0
            for im in validation_set:
                print("predicting test image ", counter)
                counter+=1
                preprocessed_color_image = self.preprocess_image(im, size)
                validation_feature_vector = []
                for feature_extractor in feature_extractors:
                    if type(feature_extractor) != SiftFeatureExtractor:
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                    else:
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(image))
                new_validation_feature_vector = clf2.transform(validation_feature_vector)
                test_prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])

            train_errors.append(train_prediction_object.evaluate(train_set_results))
            test_errors.append(test_prediction_object.evaluate(validation_set_results))

        return [train_errors, test_errors]