import os
import random
import shutil
import string
from skimage import io, transform
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
def random_augmentation(image):
    #return transform.warp(image, transform.AffineTransform(scale=[random.uniform(0.8, 1.2)]*2, rotation=random.uniform(-0.1, 0.1)))

    type = ['rotate', 'rescale', 'shear'][random.randint(0, 2)]
    if type == 'rotate':
        return transform.rotate(image, random.uniform(-5, 5))
    if type == 'rescale':
        return transform.rescale(image, random.uniform(0.5, 1.5))
    if type == 'shear':
        return transform.warp(image, transform.AffineTransform(shear=random.uniform(-0.1, 0.1)))

    return image

print("Copying training data to new folder ...")
shutil.rmtree(os.path.join(os.path.dirname(__file__), "train_augmented"))
shutil.copytree(os.path.join(os.path.dirname(__file__), "train"), os.path.join(os.path.dirname(__file__), "train_augmented"))

print("Augmenting training data ...")
for root, subFolders, files in os.walk(os.path.join(os.path.dirname(__file__), "train_augmented")):
    for file in files:
        image = io.imread(os.path.join(root, file))
        for i in range(1):
            name = os.path.join(root, file[:-4] + string.ascii_lowercase[i] + ".png")
            print("New data:", name)
            io.imsave(name, random_augmentation(image))

### VARIABLES ###
train_images_dir = os.path.join(os.path.dirname(__file__), "train_augmented")  # Train image directory
test_images_dir = os.path.join(os.path.dirname(__file__), "test_labeled")  # Test image directory
image_size = 64  # The size of the preprocessed image
nr_bins = 20  # Number of bins in the hue color histogram
radius = 64  # The radius used for calculating Zernike moments
clusters = 3  # Dominant colors used for k-means clustering before DCT
n_coeff = 1000  # Number of DCT coefficients to include in the feature vector
pixels_per_cell = 6  # Pixels per cell for HOG vector
block_size = 64 # Image size
number_of_descriptors = 250

### FEATURE EXTRACTORS ###


hog_extractor = HogFeatureExtractor(pixels_per_cell)
color_extractor = ColorFeatureExtractor(nr_bins)
shape_extractor = ShapeFeatureExtractor(radius)
symbol_extractor = SymbolFeatureExtractor(clusters, block_size, image_size)
sift_extractor = SiftFeatureExtractor()

feature_extractors = [hog_extractor]#, symbol_extractor, sift_extractor, shape_extractor]#, color_extractor]


tsr = TrafficSignRecognizer()

#tsr.make_submission(train_images_path=train_images_dir, test_images_path=test_images_dir, output_file_path="test.xlsx", feature_extractors=feature_extractors, times=2, size=64)
tsr.make_submission(train_images_path=train_images_dir, test_images_path=test_images_dir, output_file_path="test.xlsx", feature_extractors=feature_extractors, times=1, size=64)


#print(tsr.local_test(train_images_path=train_images_dir, feature_extractors=feature_extractors,k=2, nr_data_augments=1, size=64, times=1))

