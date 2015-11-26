import os
from predict.colorfeatureextractor import ColorFeatureExtractor
from predict.hogfeatureextractor import HogFeatureExtractor
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

### VARIABLES ###
train_images_dir = os.path.join(os.path.dirname(__file__), "train")  # Train image directory
test_images_dir = os.path.join(os.path.dirname(__file__), "test_labeled")  # Test image directory
image_size = 64  # The size of the preprocessed image
nr_bins = 20  # Number of bins in the hue color histogram
radius = 64  # The radius used for calculating Zernike moments
clusters = 3  # Dominant colors used for k-means clustering before DCT
n_coeff = 1000  # Number of DCT coefficients to include in the feature vector
pixels_per_cell = 8  # Pixels per cell for HOG vector
block_size = 64 # Image size
number_of_descriptors = 250

### FEATURE EXTRACTORS ###


hog_extractor = HogFeatureExtractor(pixels_per_cell)
color_extractor = ColorFeatureExtractor(nr_bins)
shape_extractor = ShapeFeatureExtractor(radius)
symbol_extractor = SymbolFeatureExtractor(clusters, block_size, image_size)
sift_extractor = SiftFeatureExtractor()

feature_extractors = [hog_extractor]#, color_extractor, shape_extractor, symbol_extractor, sift_extractor]


tsr = TrafficSignRecognizer()
tsr.make_submission(train_images_path=train_images_dir, test_images_path=test_images_dir,
                                      output_file_path="test.xlsx", feature_extractors=feature_extractors, size=64)

#print(tsr.local_test(train_images_path=train_images_dir, feature_extractors=feature_extractors,
#                     k=2, nr_data_augments=1, size=64,times=1))

