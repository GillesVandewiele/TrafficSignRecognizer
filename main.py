import os
import cv2
from numpy import append, resize
from skimage import color
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from inout.fileparser import FileParser
from predict.colorfeatureextractor import ColorFeatureExtractor
from predict.prediction import Prediction
from predict.shapefeatureextractor import ShapeFeatureExtractor
from predict.symbolfeatureextractor import SymbolFeatureExtractor

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


def preprocess_image(image):
    image_array = cv2.imread(image)
    denoised_image = cv2.fastNlMeansDenoisingColored(image_array,None,3,3,7,21)
    return [denoised_image,color.rgb2gray(denoised_image)]


train_images_dir = os.path.join(os.path.dirname(__file__), "train")
test_images_dir = os.path.join(os.path.dirname(__file__), "test")

train_images = get_images_from_directory(train_images_dir)
test_images = get_images_from_directory(test_images_dir)
results = get_results(train_images_dir)

# Training phase
feature_vectors = []
color_extractor = ColorFeatureExtractor()
shape_extractor = ShapeFeatureExtractor()
symbol_extractor = SymbolFeatureExtractor()

for image in train_images:
    print("Training ", image, "...")

    [color_image,gray_image] = preprocess_image(image)

    #feature_vector = color_extractor.extract_hog(gray_image)

    # First, calculate the Zernike moments
    feature_vector = shape_extractor.extract_zernike(gray_image)

    # Then the HOG, our most important feature(s)
    #feature_vector = color_extractor.extract_hog(image)
    feature_vector = append(feature_vector, color_extractor.extract_hog(gray_image))

    # Then we extract the color features
    hue = color_extractor.extract_hue(color_image)
    feature_vector = append(feature_vector,color_extractor.calculate_histogram(hue, 20))

    # Then we add the shape_features using the hue from the color edxtractor
    contour = shape_extractor.calculateRimContour(hue)
    shape_features = shape_extractor.calculateGeometricMoments(contour)
    feature_vector = append(feature_vector, shape_features)

    # Finally we append DCT coefficients
    feature_vector = append(feature_vector,symbol_extractor.calculateDCT(gray_image))

    # Append our feature_vector to the feature_vectors
    feature_vectors.append(feature_vector)

"""
clf = SVC(C=1.0, cache_size=3000, class_weight=None, kernel='linear', max_iter=-1, probability=True,
                  random_state=None, shrinking=False, tol=0.001, verbose=False)
"""

clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=32, intercept_scaling=1,
                         class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                         multi_class='ovr', verbose=0)

"""
# Logistic Regression for feature selection, higher C = more features will be deleted
clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4)

new_feature_vectors = clf2.fit_transform(feature_vectors, results)
"""
clf.fit(feature_vectors, results)

# Testing phase
prediction_object = Prediction()
for im in test_images:
    print("Predict: ", im)

    [color_image,gray_image] = preprocess_image(im)

    #validation_feature_vector = color_extractor.extract_hog(gray_image)


    # Calculate Zernike moments
    validation_feature_vector = shape_extractor.extract_zernike(gray_image)

    # Extract validation_feature_vector
    # validation_feature_vector = color_extractor.extract_hog(im)
    validation_feature_vector = append(validation_feature_vector, color_extractor.extract_hog(gray_image))

    # Extract the same color features as the training phase
    hue = color_extractor.extract_hue(color_image)
    validation_feature_vector = append(validation_feature_vector,color_extractor.calculate_histogram(hue, 20))

    # And the same shape features
    contour = shape_extractor.calculateRimContour(hue)
    shape_features = shape_extractor.calculateGeometricMoments(contour)
    validation_feature_vector = append(validation_feature_vector, shape_features)

    # Calculate the DCT coeffs
    validation_feature_vector = append(validation_feature_vector,symbol_extractor.calculateDCT(gray_image))

    #new_validation_feature_vector = clf2.transform(validation_feature_vector)
    #print(clf.predict_proba(new_validation_feature_vector)[0])
    prediction_object.addPrediction(clf.predict_proba(validation_feature_vector)[0])

FileParser.write_CSV("submission.xlsx", prediction_object)

"""
x = 2250
percent = 0.95
fault = 0.0005
p1 = math.log(max(min(fault, 1-pow(10, -15)), pow(10, -15)))
p2 = math.log(max(min(1-(80*fault), 1-pow(10, -15)), pow(10, -15)))
print(p1)
print(p2)
print(p1*(1-percent) + p2*(percent))
"""