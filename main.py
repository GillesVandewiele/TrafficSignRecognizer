import os
import cv2
from numpy import append, resize
from skimage import color
from sklearn.linear_model import LogisticRegression
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

train_images_dir = os.path.join(os.path.dirname(__file__), "train")
test_images_dir = os.path.join(os.path.dirname(__file__), "test")

train_images = get_images_from_directory(train_images_dir)
test_images = get_images_from_directory(test_images_dir)
results = get_results(train_images_dir)
#test_results = get_results(test_images_dir)

# Training phase
feature_vectors = []
color_extractor = ColorFeatureExtractor()
shape_extractor = ShapeFeatureExtractor()
symbol_extractor = SymbolFeatureExtractor()

def preprocess_image(image, _color=False, size=64):
    image_array = cv2.imread(image)
    denoised_image = cv2.fastNlMeansDenoisingColored(image_array,None,10,10,7,21)
    if _color:
        return resize(denoised_image, (size,size, 3))
    else:
        return resize(color.rgb2gray(denoised_image), (size,size))

for image in train_images:
    print("Training ", image, "...")

    preprocessed_color_image = preprocess_image(image, _color=True, size=64)
    preprocessed_gray_image = preprocess_image(image, _color=False, size=64)

    # First, calculate the Zernike moments
    feature_vector = shape_extractor.extract_zernike(preprocessed_gray_image, 64)

    # Then the HOG, our most important feature(s)
    feature_vector = append(feature_vector, color_extractor.extract_hog(preprocessed_gray_image))

    # Then we extract the color features
    hue = color_extractor.extract_hue(preprocessed_color_image)
    feature_vector = append(feature_vector,color_extractor.calculate_histogram(hue, 20))

    # Then we add the shape_features using the hue from the color edxtractor
    contour = shape_extractor.calculateRimContour(hue)
    shape_features = shape_extractor.calculateGeometricMoments(contour)
    feature_vector = append(feature_vector, shape_features)

    # Finally we append DCT coefficients
    feature_vector = append(feature_vector,symbol_extractor.calculateDCT(preprocessed_gray_image))

    # Append our feature_vector to the feature_vectors
    feature_vectors.append(feature_vector)

"""
clf = SVC(C=1.0, cache_size=3000, class_weight=None, kernel='linear', max_iter=-1, probability=True,
                  random_state=None, shrinking=False, tol=0.001, verbose=False)
"""

clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=32, intercept_scaling=1, solver='liblinear', max_iter=100,
                         multi_class='ovr', verbose=0)

# Logistic Regression for feature selection, higher C = more features will be deleted
clf2 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=4)

new_feature_vectors = clf2.fit_transform(feature_vectors, results)
clf.fit(new_feature_vectors, results)

# Testing phase
prediction_object = Prediction()
for im in test_images:
    print("Predicting ", im)

    preprocessed_color_image = preprocess_image(im, _color=True, size=64)
    preprocessed_gray_image = preprocess_image(im, _color=False, size=64)

    # Calculate Zernike moments
    validation_feature_vector = shape_extractor.extract_zernike(preprocessed_gray_image, 64)

    #validation_feature_vector = color_extractor.extract_hog(im)
    # Extract validation_feature_vector
    validation_feature_vector = append(validation_feature_vector, color_extractor.extract_hog(preprocessed_gray_image))

    # Extract the same color features as the training phase
    hue = color_extractor.extract_hue(preprocessed_color_image)
    validation_feature_vector = append(validation_feature_vector,color_extractor.calculate_histogram(hue, 20))

    # And the same shape features
    contour = shape_extractor.calculateRimContour(hue)
    shape_features = shape_extractor.calculateGeometricMoments(contour)
    validation_feature_vector = append(validation_feature_vector, shape_features)

    # Calculate the DCT coeffs
    validation_feature_vector = append(validation_feature_vector,symbol_extractor.calculateDCT(preprocessed_gray_image))

    new_validation_feature_vector = clf2.transform(validation_feature_vector)
    print(clf.predict_proba(new_validation_feature_vector)[0])
    prediction_object.addPrediction(clf.predict_proba(new_validation_feature_vector)[0])

#print("Logloss score = ", prediction_object.evaluate(test_results))
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