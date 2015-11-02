import os
from numpy import append, arange, delete, where
import random
from scipy.stats import rv_discrete
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

def get_training_set(nr_samples, training_set, results, seed, k):
    nr_classes = len(Prediction.TRAFFIC_SIGNS)

    if(nr_samples < k*nr_classes):
        # Fuck exceptions
        print("Number of samples should be greater than k*nr_classes")
        return

    if(nr_samples%k != 0):
        print("Number of samples should be divisible by k!")
        return

    random.seed(seed)

    # Use the benchmark predictor to calculate the occurrence of each class
    benchmark = BenchmarkPredictor()
    benchmark.train(training_set, results)

    # Divide the training set in his classes
    training_set_per_class = {}
    results_set = sorted(set(results))
    for i in range(nr_classes):
        training_set_per_class[results_set[i]] = []

    for i in range(len(training_set)):
        training_set_per_class[results[i]].append(training_set[i])

    # First add a sample of each class to the new training set
    sign_counter = 0
    new_train_set = []
    new_validation_set = []
    new_train_results = []
    new_validation_results = []
    for i in range(nr_classes):
        # Put an element in the new training set
        index = random.randint(0, len(training_set_per_class[results_set[i]])-1)
        new_train_set.append(training_set_per_class[results_set[i]][index])
        training_set_per_class[results_set[i]].pop(index)
        new_train_results.append(results_set[i])

        # And in the validation set
        index = random.randint(0, len(training_set_per_class[results_set[i]])-1)
        new_validation_set.append(training_set_per_class[results_set[i]][index])
        training_set_per_class[results_set[i]].pop(index)
        new_validation_results.append(results_set[i])

    # Now we got an equal validation and train set, so we need to add k*nr_classes using the distribution from
    # benchmark predictor
    values = arange(nr_classes)
    probs = sorted(benchmark.occurrenceProbabilities.values())
    custm = rv_discrete(values=(values, probs))
    for i in range((k-2)*nr_classes):
        sign = custm.rvs(size=1)
        while(len(training_set_per_class[results_set[sign]]) == 0):
            values = delete(values, where(values==sign))
            probs = delete(probs, where(probs==sign))
            custm = rv_discrete(values=(values, probs))
            results_set.pop(sign)
            sign = custm.rvs(size=1)

        index = random.randint(0, len(training_set_per_class[results_set[sign]])-1)
        new_train_set.append(training_set_per_class[results_set[sign]][index])
        training_set_per_class[results_set[sign]].pop(index)
        new_train_results.append(results_set[sign])

    # Now use the previous calculated distributions to assign the required amount of remaining samples to the dataset
    values = arange(nr_classes)
    probs = sorted(benchmark.occurrenceProbabilities.values())
    custm = rv_discrete(values=(values, probs))
    for i in range(int((nr_samples - k*nr_classes)/k)):

        # Add k-1 samples to the train set
        for j in range(k-1):
            sign = custm.rvs(size=1)
            while(len(training_set_per_class[results_set[sign]]) == 0):
                values = delete(values, where(values==sign))
                probs = delete(probs, where(probs==sign))
                custm = rv_discrete(values=(values, probs))
                results_set.pop(sign)
                sign = custm.rvs(size=1)

            index = random.randint(0, len(training_set_per_class[results_set[sign]])-1)
            new_train_set.append(training_set_per_class[results_set[sign]][index])
            training_set_per_class[results_set[sign]].pop(index)
            new_train_results.append(results_set[sign])

        # And 1 sample to the validation set
        sign = custm.rvs(size=1)
        while(len(training_set_per_class[results_set[sign]]) == 0):
            values = delete(values, where(values==sign))
            probs = delete(probs, where(probs==sign))
            custm = rv_discrete(values=(values, probs))
            results_set.pop(sign)
            sign = custm.rvs(size=1)

        index = random.randint(0, len(training_set_per_class[results_set[sign]])-1)
        new_validation_set.append(training_set_per_class[results_set[sign]][index])
        training_set_per_class[results_set[sign]].pop(index)
        new_validation_results.append(results_set[sign])


        i += (k-1)

    return [new_train_set, new_validation_set, new_train_results, new_validation_results]





def classify_traffic_signs(train_set,validation_set, train_set_results, validation_set_results):

    # Iterate over the training set and transform each input vector to a feature vector
    feature_vectors = []
    color_extractor = ColorFeatureExtractor()
    shape_extractor = ShapeFeatureExtractor()
    symbol_extractor = SymbolFeatureExtractor()

    for image in train_set:

        # First, calculate the Zernike moments
        feature_vector = shape_extractor.extract_zernike(image)


        # Then the HOG, our most important feature(s)
        #feature_vector = color_extractor.extract_hog(image)
        feature_vector = append(feature_vector, color_extractor.extract_hog(image))

        # Then we extract the color features
        hue = color_extractor.extract_hue(image)
        feature_vector = append(feature_vector,color_extractor.calculate_histogram(hue, 20))

        # Then we add the shape_features using the hue from the color edxtractor
        contour = shape_extractor.calculateRimContour(hue)
        shape_features = shape_extractor.calculateGeometricMoments(contour)
        feature_vector = append(feature_vector, shape_features)

        # Finally we append DCT coefficients
        feature_vector = append(feature_vector,symbol_extractor.calculateDCT(image))

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
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=16, intercept_scaling=1,
                             class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                             multi_class='ovr', verbose=0)

    # Fit the model
    clf.fit(feature_vectors, train_set_results)

    prediction_object = Prediction()
    for im in validation_set:

        # Calculate Zernike moments
        validation_feature_vector = shape_extractor.extract_zernike(im)

        #validation_feature_vector = color_extractor.extract_hog(im)
        # Extract validation_feature_vector
        validation_feature_vector = append(validation_feature_vector, color_extractor.extract_hog(im))

        # Extract the same color features as the training phase
        hue = color_extractor.extract_hue(im)
        validation_feature_vector = append(validation_feature_vector,color_extractor.calculate_histogram(hue, 20))

        # And the same shape features
        contour = shape_extractor.calculateRimContour(hue)
        shape_features = shape_extractor.calculateGeometricMoments(contour)
        validation_feature_vector = append(validation_feature_vector, shape_features)

        # Calculate the DCT coeffs
        validation_feature_vector = append(validation_feature_vector,symbol_extractor.calculateDCT(im))

        #print(clf.predict(validation_feature_vector)[0])

        prediction_object.addPrediction(clf.predict_proba(validation_feature_vector)[0])

    return prediction_object.evaluate(validation_set_results)


train_images_dir = os.path.join(os.path.dirname(__file__), "train")
train_images = get_images_from_directory(train_images_dir)
results = get_results(train_images_dir)

sizes = [256, 512, 1024, 2048]

for size in sizes:
    print("Calculating the logloss for size: ", size)
    new_train_set, new_validation_set, new_train_set_results, new_validation_set_results = get_training_set(size, train_images, results, 1337, 2)
    score1 = classify_traffic_signs(new_train_set, new_validation_set, new_train_set_results, new_validation_set_results)
    score2 = classify_traffic_signs(new_validation_set, new_train_set, new_validation_set_results, new_train_set_results)
    print("Avg score using a dataset of size ", size, " = ", (score1+score2)/2)

# Or use
print("Calculating the logloss for size: ", len(train_images))
kf = KFold(len(train_images), n_folds=2, shuffle=True, random_state=1337)
scores = []
for train, validation in kf:
        # Divide the train_images in a training and validation set (using KFold)
        train_set = [train_images[i] for i in train]
        validation_set = [train_images[i] for i in validation]
        train_set_results = [results[i] for i in train]
        validation_set_results = [results[i] for i in validation]
        scores.append(classify_traffic_signs(train_set, validation_set, train_set_results, validation_set_results))
print("Avg score using a dataset of size ", len(train_images), " = ", sum(scores)/len(scores))

