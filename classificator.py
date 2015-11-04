import os
import cv2
from numpy import append, arange, delete, where, resize
import random
from scipy.stats import rv_discrete
from skimage import color
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from predict.benchmark import BenchmarkPredictor
from predict.colorfeatureextractor import ColorFeatureExtractor
from predict.prediction import Prediction
from predict.shapefeatureextractor import ShapeFeatureExtractor
from predict.symbolfeatureextractor import SymbolFeatureExtractor

__author__ = 'Group16'

"""
    Main class of our project. Use cross-validation to divide our training set into a new training set and a validation
    set. Then we construct a feature vector and use Logistic Regression or SVM to construct a model and later on
    predict the elements from the validation set. Finally, a logloss score is calculated.
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

"""
#TODO: divide in 4 classes:
def transform_classes(results):
    # Reduce the number of classes in the results for color classification, the resulting classes are:
    #   red signs (red_circles, stop, forbidden, others: F41)
    #   semi-red signs (triangles, red_blue_circles, reversed triangles, others: F43)
    #   blue signs (blue_circles, rectangles_down, rectangles_up, squares, others: F13, F21, F23A, F25, F27, F29, Handic)
    #   white signs (other: C37, F1, F1a_h, F33_34, F3a_h, F4b, begin, end, e0c, lang, m)
    #   others (diamonds, others: F31, F35)
    new_classes = []
    for result in results:
        if result in ["B19", "C3", "C11", "C21", "C23", "C29", "C31", "C35", "C43", "F4a", "F41", "C1", "B5"]:
                     # "B1", "B3", "B7", "E1", "E5", "E3", "E7", "A1AB", "A1CD", "A7A", "A7B", "A13", "A14", "A15",
                     # "A23", "A25", "A29", "A31", "A51", "B15A", "B17", "F43"]:
            new_classes.append("red")
        elif result in ["B1", "B3", "B7", "E1", "E5", "E3", "E7", "A1AB", "A1CD", "A7A", "A7B", "A13", "A14", "A15",
                        "A23", "A25", "A29", "A31", "A51", "B15A", "B17", "F43"]:
            new_classes.append("semi-red")
        elif result in ["D1a", "D1b", "D1e", "D5", "D7", "D9", "D10", "F12a", "F12b", "B21", "E9a", "E9a_miva", "E9b",
                        "E9cd", "E9e", "F45", "F47", "F59", "X", "F19", "F49", "F50", "F87", "F13", "F21", "F23A",
                        "F25", "F27", "F29", "Handic"]:
            new_classes.append("blue")
        #elif result in ["C37", "F1", "F1a_h", "F33_34", "F3a_h", "F4b", "begin", "end", "e0c", "lang", "m"]:
        #    new_classes.append("white")
        else:
            new_classes.append("others")
    return new_classes
"""

from itertools import chain
def flatten_dict_values(dictionary):
     return set(chain(*dictionary.values()))

def get_training_set(nr_samples, training_set, results, seed):
    nr_classes = len(Prediction.TRAFFIC_SIGNS)

    if(nr_samples < 2*nr_classes):
        # Fuck exceptions
        print("Number of samples should be greater than 2*nr_classes")
        return

    if(nr_samples%2 != 0):
        print("Number of samples should be divisible by 2!")
        return

    random.seed(seed)

    # Use the benchmark predictor to calculate the occurrence of each class
    benchmark = BenchmarkPredictor()
    benchmark.train(training_set, results)

    # Divide the training set in his classes
    training_set_per_class = {}
    results_set_per_class = {}
    results_set = sorted(set(results))
    for i in range(len(results_set)):
        training_set_per_class[results_set[i]] = []
        results_set_per_class[results_set[i]] = []

    for i in range(len(training_set)):
        training_set_per_class[results[i]].append(training_set[i])
        results_set_per_class[results[i]].append(results[i])

    # First add a sample of each class to the new training set
    new_train_set = []
    new_validation_set = []
    new_train_results = []
    new_validation_results = []
    remaining_samples = nr_samples
    if(min([len(training_set_per_class[i]) for i in training_set_per_class]) > 1):
        for i in range(len(results_set)):
            # Put an element in the new training set
            index = random.randint(0, len(training_set_per_class[results_set[i]])-1)
            new_train_set.append(training_set_per_class[results_set[i]][index])
            training_set_per_class[results_set[i]].pop(index)
            results_set_per_class[results_set[i]].pop(index)
            new_train_results.append(results_set[i])

            # And in the validation set
            index = random.randint(0, len(training_set_per_class[results_set[i]])-1)
            new_validation_set.append(training_set_per_class[results_set[i]][index])
            training_set_per_class[results_set[i]].pop(index)
            results_set_per_class[results_set[i]].pop(index)
            new_validation_results.append(results_set[i])

        remaining_samples = nr_samples-2*len(results_set)

    # Now use the previous calculated distributions to assign the required amount of remaining samples to the dataset
    values = arange(len(results_set))
    probs = sorted(benchmark.occurrenceProbabilities.values())
    custm = rv_discrete(values=(values, probs))
    results_set = sorted(flatten_dict_values(results_set_per_class))
    for j in range(int(remaining_samples/2)):
        sign = custm.rvs(size=1)
        while(sign >= len(results_set) or results_set[sign] not in training_set_per_class or len(training_set_per_class[results_set[sign]]) == 0):
            values = delete(values, where(values==sign))
            probs = delete(probs, where(probs==sign))
            custm = rv_discrete(values=(values, probs))
            if(sign < len(results_set)):
                results_set.pop(sign)
            sign = custm.rvs(size=1)

        index = random.randint(0, len(training_set_per_class[results_set[sign]])-1)
        new_train_set.append(training_set_per_class[results_set[sign]][index])
        training_set_per_class[results_set[sign]].pop(index)
        results_set_per_class[results_set[sign]].pop(index)
        new_train_results.append(results_set[sign])

    values = arange(len(results_set))
    probs = sorted(benchmark.occurrenceProbabilities.values())
    custm = rv_discrete(values=(values, probs))
    results_set = sorted(flatten_dict_values(results_set_per_class))
    for j in range(int(remaining_samples/2)):
        # And 1 sample to the validation set
        sign = custm.rvs(size=1)
        while(sign >= len(results_set) or results_set[sign] not in training_set_per_class or len(training_set_per_class[results_set[sign]]) == 0):
            values = delete(values, where(values==sign))
            probs = delete(probs, where(probs==sign))
            custm = rv_discrete(values=(values, probs))
            if(sign < len(results_set)):
                results_set.pop(sign)
            sign = custm.rvs(size=1)

        index = random.randint(0, len(training_set_per_class[results_set[sign]])-1)
        new_validation_set.append(training_set_per_class[results_set[sign]][index])
        training_set_per_class[results_set[sign]].pop(index)
        results_set_per_class[results_set[sign]].pop(index)
        new_validation_results.append(results_set[sign])

    return [new_train_set, new_validation_set, new_train_results, new_validation_results]


def preprocess_image(image):
    image_array = cv2.imread(image)
    denoised_image = cv2.fastNlMeansDenoisingColored(image_array,None,3,3,7,21)
    return color.rgb2gray(denoised_image)



def classify_traffic_signs(train_set,validation_set, train_set_results, validation_set_results):

    # Iterate over the training set and transform each input vector to a feature vector
    feature_vectors = []
    color_extractor = ColorFeatureExtractor()
    shape_extractor = ShapeFeatureExtractor()
    symbol_extractor = SymbolFeatureExtractor()

    counter = 0
    for image in train_set:

        print("Training image ", image)
        counter+=1

        # First, calculate the Zernike moments
        #feature_vector = shape_extractor.extract_zernike(image)


        # Then the HOG, our most important feature(s)
        feature_vector = color_extractor.extract_hog(preprocess_image(image))
        #feature_vector = append(feature_vector, color_extractor.extract_hog(image))

        # Then we extract the color features
        #hue = color_extractor.extract_hue(image)
        #feature_vector = append(feature_vector,color_extractor.calculate_histogram(hue, 20))

        # Then we add the shape_features using the hue from the color edxtractor
        #contour = shape_extractor.calculateRimContour(hue)
        #shape_features = shape_extractor.calculateGeometricMoments(contour)
        #feature_vector = append(feature_vector, shape_features)

        # Finally we append DCT coefficients
        #feature_vector = append(feature_vector,symbol_extractor.calculateDCT(image))


        # Append our feature_vector to the feature_vectors
        feature_vectors.append(feature_vector)

    # We use C-SVM with a linear kernel and want to predict probabilities
    # max_iter = -1 for no limit on iterations (tol is our stopping criterion)
    # Put verbose off for some output and don't use the shrinking heuristic (needs some testing)
    # Allocate 1 GB of memory for our kernel
    # We are using seed 1337 to always get the same results (can be put on None for testing)
    """
    clf = SVC(C=1.0, cache_size=3000, class_weight=None, kernel='linear', max_iter=-1, probability=True,
              random_state=1337, shrinking=False, tol=0.0001, verbose=False)
    """

    # Use Logistic Regression instead of SVM
    clf = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4, intercept_scaling=1,
                             class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                             multi_class='ovr', verbose=0)


    # Logistic Regression for feature selection, higher C = more features will be deleted
    clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.5)

    #Print feature vector length
    print(len(feature_vectors), len(feature_vectors[0]))
    new_feature_vectors = clf2.fit_transform(feature_vectors, train_set_results)
    print(new_feature_vectors.shape)

    # Fit the model
    clf.fit(new_feature_vectors, train_set_results)
    """
    train_prediction_object = Prediction()
    counter=0
    for im in train_set:

        #print("predicting training image ", counter)
        counter+=1

        # Calculate Zernike moments
        #validation_feature_vector = shape_extractor.extract_zernike(im)

        # Extract validation_feature_vector
        validation_feature_vector = color_extractor.extract_hog(color.rgb2gray(cv2.imread(im)))
        #validation_feature_vector = append(validation_feature_vector, color_extractor.extract_hog(im))

        # Extract the same color features as the training phase
        #hue = color_extractor.extract_hue(im)
        #validation_feature_vector = append(validation_feature_vector,color_extractor.calculate_histogram(hue, 20))

        # And the same shape features
        #contour = shape_extractor.calculateRimContour(hue)
        #shape_features = shape_extractor.calculateGeometricMoments(contour)
        #validation_feature_vector = append(validation_feature_vector, shape_features)

        # Calculate the DCT coeffs
        #validation_feature_vector = append(validation_feature_vector,symbol_extractor.calculateDCT(im))

        #print(clf.predict(validation_feature_vector)[0])

        new_validation_feature_vector = clf2.transform(validation_feature_vector)
        train_prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])
    """
    validation_prediction_object = Prediction()
    counter=0
    for im in validation_set:

        print("predicting test image ", im)
        counter+=1

        # Calculate Zernike moments
        #validation_feature_vector = shape_extractor.extract_zernike(im)

        # Extract validation_feature_vector
        validation_feature_vector = color_extractor.extract_hog(preprocess_image(im))
        #validation_feature_vector = append(validation_feature_vector, color_extractor.extract_hog(im))

        # Extract the same color features as the training phase
        #hue = color_extractor.extract_hue(im)
        #validation_feature_vector = append(validation_feature_vector,color_extractor.calculate_histogram(hue, 20))

        # And the same shape features
        #contour = shape_extractor.calculateRimContour(hue)
        #shape_features = shape_extractor.calculateGeometricMoments(contour)
        #validation_feature_vector = append(validation_feature_vector, shape_features)

        # Calculate the DCT coeffs
        #validation_feature_vector = append(validation_feature_vector,symbol_extractor.calculateDCT(im))

        #print(clf.predict(validation_feature_vector)[0])

        new_validation_feature_vector = clf2.transform(validation_feature_vector)
        validation_prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])

    return [validation_prediction_object.evaluate(validation_set_results),0]
    #return [validation_prediction_object.evaluate(validation_set_results), train_prediction_object.evaluate(train_set_results)]


train_images_dir = os.path.join(os.path.dirname(__file__), "train")
train_images = get_images_from_directory(train_images_dir)
all_train_images = []
for image in train_images:
    all_train_images.append(image)
results = get_results(train_images_dir)
all_results = []
for result in results:
    all_results.append(result)
"""
print(len(train_images))
sizes = [4000]

new_train_set = []
new_validation_set = []
new_train_set_results = []
new_validation_set_results = []

for size in sizes:
    new_train_set_temp, new_validation_set_temp, new_train_set_results_temp, new_validation_set_results_temp = get_training_set(size, train_images, results, 1337)
    new_train_set.extend(new_train_set_temp)
    new_validation_set.extend(new_validation_set_temp)
    new_train_set_results.extend(new_train_set_results_temp)
    new_validation_set_results.extend(new_validation_set_results_temp)
    print("Calculating the logloss for size: ", len(new_train_set)+len(new_validation_set))
    [validation_score1, train_score1] = classify_traffic_signs(new_train_set, new_validation_set, new_train_set_results, new_validation_set_results)
    [validation_score2, train_score2] = classify_traffic_signs(new_validation_set, new_train_set, new_validation_set_results, new_train_set_results)
    for i in range(len(new_train_set_temp)):
        train_images.pop(train_images.index(new_train_set_temp[i]))
        results.pop(results.index(new_train_set_results_temp[i]))
        train_images.pop(train_images.index(new_validation_set_temp[i]))
        results.pop(results.index(new_validation_set_results_temp[i]))
    print("Avg training score using a dataset of size ", len(new_train_set)+len(new_validation_set), " = ", (train_score1+train_score2)/2)
    print("Avg validation score using a dataset of size ", len(new_train_set)+len(new_validation_set), " = ", (validation_score1+validation_score2)/2)


"""
# Or use
print("Calculating the logloss for size: ", len(all_train_images))
kf = KFold(len(train_images), n_folds=2, shuffle=True, random_state=1337)
validation_scores = []
train_scores = []
# all_train_images = all_train_images[600:900]
# all_results = all_results[600:900]
for train, validation in kf:
        # Divide the train_images in a training and validation set (using KFold)
        train_set = [all_train_images[i] for i in train]
        validation_set = [all_train_images[i] for i in validation]
        train_set_results = [all_results[i] for i in train]
        validation_set_results = [all_results[i] for i in validation]
        validation_score, train_score = classify_traffic_signs(train_set, validation_set, train_set_results, validation_set_results)
        validation_scores.append(validation_score)
        train_scores.append(train_score)
print("Avg training score using a dataset of size ", len(all_train_images), " = ", sum(train_scores)/len(train_scores))
print("Avg validation score using a dataset of size ", len(all_train_images), " = ", sum(validation_scores)/len(validation_scores))
