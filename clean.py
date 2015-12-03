import cv2
from lasagne import layers
import lasagne
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from numpy import append
from skimage.transform import resize
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from inout.fileparser import FileParser
from predict.prediction import Prediction
from predict.siftfeatureextractor import SiftFeatureExtractor
import numpy as np

class Recognizer(object):

    def preprocess_image(self, image, size):
        image_array = cv2.imread(image)
        image_array = resize(image_array, (size, size, 3))
        return image_array

    def build_nn(self, nr_features):
        net1 = NeuralNet(
            layers=[  # three layers: one hidden layer
                ('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('hidden2', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
            # layer parameters:1152 576
            input_shape=(None, nr_features),  # 96x96 input pixels per batch
            hidden_num_units=1152,  # number of units in hidden layer
            hidden2_num_units=576,  # number of units in hidden layer
            output_nonlinearity=lasagne.nonlinearities.softmax,  # output layer uses identity function
            output_num_units=81,  # 30 target values

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=300,
            verbose=1,
        )
        return net1

    def build_conv(self):
        net1 = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('convlayer1', layers.Conv2DLayer),
                ('poollayer1', layers.MaxPool2DLayer),
                ('convlayer2', layers.Conv2DLayer),
                ('poollayer2', layers.MaxPool2DLayer),
                ('convlayer3', layers.Conv2DLayer),
                ('poollayer3', layers.MaxPool2DLayer),
                ('dense1', layers.DenseLayer),
                ('dropout', layers.DropoutLayer),
                ('output', layers.DenseLayer)
            ],
            input_shape=(None, 3, 48, 48),
            convlayer1_num_filters = 32, convlayer1_filter_size=(7,7),
            poollayer1_pool_size=(2,2),
            convlayer2_num_filters = 64, convlayer2_filter_size=(4,4),
            poollayer2_pool_size=(2,2),
            convlayer3_num_filters = 4, convlayer3_filter_size=(4,4),
            poollayer3_pool_size=(2,2),
            dense1_num_units=300,
            output_nonlinearity=lasagne.nonlinearities.softmax,  # output layer uses identity function
            output_num_units=81,  # 30 target values

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=100,  # we want to train this many epochs
            verbose=2,
        )
        return net1

    def make_submission(self, train_images, train_results, test_images, output_file_path, feature_extractors, model, size=64):
        # Create a vector of feature vectors and initialize the codebook of sift extractor
        feature_vectors = []
        sift_extractor = temp_extractor = next((extractor for extractor in feature_extractors if type(extractor) == SiftFeatureExtractor), None)
        if(sift_extractor != None):
            sift_extractor.set_codebook(train_images)
            feature_extractors[feature_extractors.index(temp_extractor)] = sift_extractor

        # Extract features from every image
        for image in train_images:
            print("Training ", image, "...")
            preprocessed_color_image = self.preprocess_image(image, size)
            feature_vector = []
            if feature_extractors != []:
                for feature_extractor in feature_extractors:
                    if type(feature_extractor) != SiftFeatureExtractor:
                        feature_vector = append(feature_vector,
                                                feature_extractor.extract_feature_vector(preprocessed_color_image))
                    else:
                        feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(image))
            else:
                feature_vector = np.asarray(resize(cv2.imread(image), (48, 48, 3)).transpose(2,0,1).reshape(3, 48, 48))
            feature_vectors.append(feature_vector)

        # Logistic Regression for feature selection, higher C = more features will be deleted
        clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4)

        # Feature selection/reduction
        if(model != "conv"):
            new_feature_vectors = clf2.fit_transform(feature_vectors, train_results)
            if(model == "neural"):
                model = self.build_nn(nr_features=len(new_feature_vectors[0]))

            # Fit our model
            model.fit(new_feature_vectors, train_results)

        else:
            model = self.build_conv()

            # Fit our model
            model.fit(feature_vectors, train_results)

        # Iterate over the test images and add their prediction to a prediction object
        prediction_object = Prediction()
        for im in test_images:
            print("Predicting ", im)
            preprocessed_color_image = self.preprocess_image(im, size)
            validation_feature_vector = []
            if feature_extractors != []:
                for feature_extractor in feature_extractors:
                    if type(feature_extractor) != SiftFeatureExtractor:
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                    else:
                        validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(im))
                validation_feature_vector = clf2.transform(validation_feature_vector)
            else:
                validation_feature_vector = np.asarray(resize(cv2.imread(image), (48, 48, 3)).transpose(2,0,1).reshape(3, 48, 48))
            prediction_object.addPrediction(model.predict_proba(validation_feature_vector)[0])

        # Write out the prediction object
        FileParser.write_CSV(output_file_path, prediction_object)

    def local_test(self, images, results, feature_extractors, model, k=2, size=64):
        kf = KFold(len(images), n_folds=k, shuffle=True, random_state=1337)
        # kf = KFold(500, n_folds=k, shuffle=True, random_state=1337)
        train_errors = []
        test_errors = []

        for train, validation in kf:
            # Divide the train_images in a training and validation set (using KFold)
            train_set = [images[i % len(images)] for i in train]
            validation_set = [images[i % len(images)] for i in validation]
            train_set_results = [results[i % len(images)] for i in train]
            validation_set_results = [results[i % len(images)] for i in validation]

            # Create an empty feature_vectors array and set the codebook of the sift extractor if there is any
            feature_vectors = []
            sift_extractor = temp_extractor = next(
                (extractor for extractor in feature_extractors if type(extractor) == SiftFeatureExtractor), None)
            if (sift_extractor != None):
                sift_extractor.set_codebook(train_set)
                feature_extractors[feature_extractors.index(temp_extractor)] = sift_extractor

            # Iterate over the train_set, extract the features from each image and append them to feature_vectors
            for image in train_set:
                print("Training ", image, "...")
                preprocessed_color_image = self.preprocess_image(image, size)
                feature_vector = []
                if feature_extractors != []:
                    for feature_extractor in feature_extractors:
                        if type(feature_extractor) != SiftFeatureExtractor:
                            feature_vector = append(feature_vector,
                                                    feature_extractor.extract_feature_vector(preprocessed_color_image))
                        else:
                            feature_vector = append(feature_vector, feature_extractor.extract_feature_vector(image))
                else:
                    feature_vector = np.asarray(resize(cv2.imread(image), (48, 48, 3)).transpose(2,0,1).reshape(3, 48, 48))
                feature_vectors.append(feature_vector)

            # Logistic Regression for feature selection, higher C = more features will be deleted

            clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4)

            # Feature selection/reduction
            if(model != "conv"):
                new_feature_vectors = clf2.fit_transform(feature_vectors, train_set_results)
                if(model == "neural"):
                    model = self.build_nn(nr_features=len(new_feature_vectors[0]))

                # Fit our model
                model.fit(new_feature_vectors, train_set_results)

            else:
                model = self.build_conv()

                # Fit our model
                model.fit(feature_vectors, train_set_results)

            train_prediction_object = Prediction()
            counter=0
            for im in train_set:
                print("predicting train image ", counter)
                counter+=1
                preprocessed_color_image = self.preprocess_image(im, size)
                validation_feature_vector = []
                if feature_extractors != []:
                    for feature_extractor in feature_extractors:
                        if type(feature_extractor) != SiftFeatureExtractor:
                            validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                        else:
                            validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(im))
                    validation_feature_vector = clf2.transform(validation_feature_vector)
                else:
                    validation_feature_vector = np.asarray(resize(cv2.imread(image), (48, 48, 3)).transpose(2,0,1).reshape(3, 48, 48))
                train_prediction_object.addPrediction(model.predict_proba(validation_feature_vector)[0])

            print("predicting test images")
            test_prediction_object = Prediction()
            counter=0
            for im in validation_set:
                print("predicting test image ", counter)
                counter+=1
                preprocessed_color_image = self.preprocess_image(im, size)
                validation_feature_vector = []
                if feature_extractors != []:
                    for feature_extractor in feature_extractors:
                        if type(feature_extractor) != SiftFeatureExtractor:
                            validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(preprocessed_color_image))
                        else:
                            validation_feature_vector = append(validation_feature_vector, feature_extractor.extract_feature_vector(im))
                    validation_feature_vector = clf2.transform(validation_feature_vector)
                else:
                    validation_feature_vector = np.asarray(resize(cv2.imread(image), (48, 48, 3)).transpose(2,0,1).reshape(3, 48, 48))
                test_prediction_object.addPrediction(model.predict_proba(validation_feature_vector)[0])

            train_errors.append(train_prediction_object.evaluate(train_set_results))
            test_errors.append(test_prediction_object.evaluate(validation_set_results))

        return [train_errors, test_errors]