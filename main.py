import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from clean import Recognizer
from predict.colorfeatureextractor import ColorFeatureExtractor
from predict.hogfeatureextractor import HogFeatureExtractor
from predict.prediction import Prediction
from predict.shapefeatureextractor import ShapeFeatureExtractor
from predict.siftfeatureextractor import SiftFeatureExtractor
from predict.symbolfeatureextractor import SymbolFeatureExtractor
from trafficsignrecognizer import TrafficSignRecognizer

__author__ = 'Group16'

"""
    Python script using our Traffic Sign Recognizer
    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning
"""

def get_images_from_directory(directory):
    # Get all files from a directory
    images = []
    for root, subFolders, files in os.walk(directory):
        for file in files:
            images.append(os.path.join(root, file))

    return images

def get_results(train_images_dir):
    # Check all files in the directory, the parent directory of a photo is the label
    results = []
    for shapesDirectory in os.listdir(train_images_dir):
        os.listdir(os.path.join(train_images_dir, shapesDirectory))
        for signDirectory in os.listdir(os.path.join(train_images_dir, shapesDirectory)):
            results.extend([signDirectory]*len(os.listdir(os.path.join(train_images_dir, shapesDirectory, signDirectory))))

    return results

def get_results_nn(train_images_dir):
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

def extract_subset(images_path, subset_length, seed):
    all_images = []
    all_results = []
    images_subset = []
    results_subset = []
    random.seed(seed)
    # First take a random image of each class
    for shapesDirectory in os.listdir(images_path):
        os.listdir(os.path.join(images_path, shapesDirectory))
        for signDirectory in os.listdir(os.path.join(images_path, shapesDirectory)):
            index = random.randint(0, len(os.listdir(os.path.join(images_path, shapesDirectory, signDirectory))) - 1)
            images_subset.append(os.path.join(images_path, shapesDirectory, signDirectory, os.listdir(os.path.join(images_path, shapesDirectory, signDirectory))[index]))
            results_subset.append(signDirectory)

    # Now shuffle the list and take the remaining amount of required images
    for shapesDirectory in os.listdir(images_path):
        os.listdir(os.path.join(images_path, shapesDirectory))
        for signDirectory in os.listdir(os.path.join(images_path, shapesDirectory)):
            for image in os.listdir(os.path.join(images_path, shapesDirectory, signDirectory)):
                all_images.append(os.path.join(images_path, shapesDirectory, signDirectory, image))
                all_results.append(signDirectory)

    combined = list(zip(all_images, all_results))
    random.shuffle(combined)
    shuffled_images, shuffled_results = zip(*combined)

    images_subset.extend(shuffled_images[:subset_length-81])
    results_subset.extend(shuffled_results[:subset_length-81])

    return [images_subset, results_subset]

### VARIABLES ###
train_images_dir = os.path.join(os.path.dirname(__file__), "train")  # Train image directory
test_images_dir = os.path.join(os.path.dirname(__file__), "test_labeled")  # Test image directory
image_size = 64  # The size of the preprocessed image
nr_bins = 20  # Number of bins in the hue color histogram
radius = 64  # The radius used for calculating Zernike moments
clusters = 3  # Dominant colors used for k-means clustering before DCT
n_coeff = 1000  # Number of DCT coefficients to include in the feature vector
pixels_per_cell = 4  # Pixels per cell for HOG vector
block_size = 64 # Image size
number_of_descriptors = 250

### FEATURE EXTRACTORS ###
hog_extractor = HogFeatureExtractor(pixels_per_cell)
color_extractor = ColorFeatureExtractor(nr_bins)
shape_extractor = ShapeFeatureExtractor(radius)
symbol_extractor = SymbolFeatureExtractor(clusters, block_size, image_size)
sift_extractor = SiftFeatureExtractor()

feature_extractors = [hog_extractor]#, symbol_extractor, sift_extractor, shape_extractor]#, color_extractor]

### MODELS ###
linear = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=32, intercept_scaling=1, solver='liblinear', max_iter=100,
                            multi_class='ovr', verbose=0)
random_tree = RandomForestClassifier(n_estimators=100,max_features="log2",max_depth=15)
neural_network = "neural"
conv_network = "conv"  # IMPORTANT!! make sure feature_extractors = [] when passing along conv_network with our methods

tsr = Recognizer()

#train_images, train_results = extract_subset(train_images_dir, 2000, 1337)
train_images = get_images_from_directory(train_images_dir)
#train_results = get_results(train_images_dir)
train_results = get_results_nn(train_images_dir)
test_images = get_images_from_directory(test_images_dir)
print(tsr.local_test(train_images, train_results, feature_extractors, "neural"))
