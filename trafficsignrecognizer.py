import os
from random import shuffle, randint
import cv2
from numpy import append, random
from skimage.transform import resize, rotate
from skimage import transform
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

    def extract_subset(self, images_path, subset_length):
        all_images = []
        all_results = []
        images_subset = []
        results_subset = []
        # First take a random image of each class
        for shapesDirectory in os.listdir(images_path):
            os.listdir(os.path.join(images_path, shapesDirectory))
            for signDirectory in os.listdir(os.path.join(images_path, shapesDirectory)):
                index = randint(0, len(os.listdir(os.path.join(images_path, shapesDirectory, signDirectory))) - 1)
                images_subset.append(os.path.join(images_path, shapesDirectory, signDirectory,os.listdir(os.path.join(images_path, shapesDirectory, signDirectory))[index]))
                #images_subset.append(os.listdir(os.path.join(images_path, shapesDirectory, signDirectory))[index])
                results_subset.append(signDirectory)

        # Now shuffle the list and take the remaining amount of required images
        for shapesDirectory in os.listdir(images_path):
            os.listdir(os.path.join(images_path, shapesDirectory))
            for signDirectory in os.listdir(os.path.join(images_path, shapesDirectory)):
                for image in os.listdir(os.path.join(images_path, shapesDirectory, signDirectory)):
                    all_images.append(os.path.join(images_path, shapesDirectory, signDirectory,image))
                    all_results.append(signDirectory)

        print(len(all_images),len(all_results))
        combined = list(zip(all_images, all_results))
        print(len(combined))
        shuffle(combined)
        shuffled_images, shuffled_results = zip(*combined)
        print(len(shuffled_images),len(shuffled_results))

        images_subset.extend(shuffled_images[:subset_length-81])
        results_subset.extend(shuffled_results[:subset_length-81])

        print(len(images_subset),len(results_subset))

        return [images_subset, results_subset]


    def preprocess_image(self, image, size):
        # Preprocess our image, creating a denoised color and gray image resized to a square size "size"
        image_array = cv2.imread(image)
        # cv2.imshow("Image",image_array)
        # cv2.waitKey(0)

        # rotated = rotate(image_array,5)
        img = resize(image_array, (size, size, 3))
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)
        #denoised_image = cv2.fastNlMeansDenoisingColored(img,None,7,7,11,21)

        return img

    def make_submission(self, train_images_path, test_images_path, output_file_path, feature_extractors, times, size=64):
        # Extract the train images with corresponding results & the test images
        train_images = self.get_images_from_directory(train_images_path)
        results = self.get_results(train_images_path)
        test_images = self.get_images_from_directory(test_images_path)
        test_results = self.get_results(test_images_path)

        for i in range(1,times):
            train_images = train_images + train_images
            results = results + results

        # Create a vector of feature vectors (a feature matrix)
        feature_vectors = []
        sift_extractor = temp_extractor = next((extractor for extractor in feature_extractors if type(extractor) == SiftFeatureExtractor), None)
        if(sift_extractor != None):
            sift_extractor.set_codebook(train_images)
            feature_extractors[feature_extractors.index(temp_extractor)] = sift_extractor
        for image in train_images:
            #print("Training ", image, "...")
            preprocessed_color_image = self.preprocess_image(image, size)
            feature_vector = []
            for feature_extractor in feature_extractors:
                if type(feature_extractor) != SiftFeatureExtractor:
                    feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                else:
                    feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(image))
            feature_vectors.append(feature_vector)

        print("Feature vector length: ", feature_vectors[0].shape)
        # Using logistic regression as linear model to fit our feature_vectors to our results
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=32, intercept_scaling=1, solver='liblinear', max_iter=100,
                         multi_class='ovr', verbose=0)

        #clf = RandomForestClassifier(n_estimators=800, max_features="sqrt", min_samples_leaf=1, n_jobs=-1)

        # Logistic Regression for feature selection, higher C = more features will be deleted
        clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4)

        # Feature selection/reduction
        new_feature_vectors = clf2.fit_transform(feature_vectors, results)

        print("Feature reduction vector length: ", new_feature_vectors[0].shape)

        # Model fitting
        clf.fit(new_feature_vectors, results)

        # Iterate over the test images and add their prediction to a prediction object
        prediction_object = Prediction()
        for im in test_images:
            #print("Predicting ", im)
            preprocessed_color_image = self.preprocess_image(im, size)
            validation_feature_vector = []
            for feature_extractor in feature_extractors:
                if type(feature_extractor) != SiftFeatureExtractor:
                    validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                else:
                    validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(im))
            new_validation_feature_vector = clf2.transform(validation_feature_vector)
            prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])

        # Write out the prediction object
        #FileParser.write_CSV(output_file_path, prediction_object)
        print("Logloss score = ", prediction_object.evaluate(test_results))




    def local_test(self, train_images_path, feature_extractors, k=2, nr_data_augments=1, size=64, times=1):
        # Extract the train images with corresponding results
        #train_images = self.get_images_from_directory(train_images_path)
        #train_images = train_images[600:1100]
        #results = self.get_results(train_images_path)
        train_images,results = self.extract_subset(train_images_path,512)

        #results = results[600:1100]

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

            print("Training images")
            # Create a vector of feature vectors (a feature matrix)
            feature_vectors = []
            counter=1
            sift_extractor = temp_extractor = next((extractor for extractor in feature_extractors if type(extractor) == SiftFeatureExtractor), None)
            if(sift_extractor != None):
                sift_extractor.set_codebook(train_set)
                feature_extractors[feature_extractors.index(temp_extractor)] = sift_extractor
            for image in train_set:
                #print("Training image ", image)
                counter += 1
                preprocessed_color_image = self.preprocess_image(image, size)
                feature_vector = []
                for feature_extractor in feature_extractors:
                    if type(feature_extractor) != SiftFeatureExtractor:
                        feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                    else:
                        feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(image))
                feature_vectors.append(feature_vector)

            print("fitting model")
            # Using logistic regression as linear model to fit our feature_vectors to our results
            clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=32, intercept_scaling=1, solver='liblinear', max_iter=100,
                            multi_class='ovr', verbose=0)

            #clf = RandomForestClassifier(n_estimators=100,max_features="log2",max_depth=15)

            # Logistic Regression for feature selection, higher C = more features will be deleted
            clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4)

            # Feature selection/reduction
            new_feature_vectors = clf2.fit_transform(feature_vectors, train_set_results)

            #clf.coef_ = reduction.coef_
            # Model fitting
            clf.fit(new_feature_vectors, train_set_results)

            print("predicting train images")
            train_prediction_object = Prediction()
            counter=0
            for im in train_set:
                #print("predicting train image ", counter)
                counter+=1
                preprocessed_color_image = self.preprocess_image(im, size)
                validation_feature_vector = []
                for feature_extractor in feature_extractors:
                    if type(feature_extractor) != SiftFeatureExtractor:
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                    else:
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(im))
                new_validation_feature_vector = clf2.transform(validation_feature_vector)
                train_prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])

            print("predicting test images")
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
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(im))
                new_validation_feature_vector = clf2.transform(validation_feature_vector)
                test_prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])

            train_errors.append(train_prediction_object.evaluate(train_set_results))
            test_errors.append(test_prediction_object.evaluate(validation_set_results))

        return [train_errors, test_errors]