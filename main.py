import os
from numpy import append
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

train_images_dir = os.path.join(os.path.dirname(__file__), "train")
test_images_dir = os.path.join(os.path.dirname(__file__), "test")

train_images = get_images_from_directory(train_images_dir)
test_images = get_images_from_directory(test_images_dir)
results = get_results(train_images_dir)

# Training phase
feature_vectors = []
color_extractor = ColorFeatureExtractor()
shape_extractor = ShapeFeatureExtractor()

for image in train_images:
    print("Train: ", image)
    feature_vector = color_extractor.extract_hog(image)
    #print(len(feature_vector))

    # First we extract the color features
    #hue = color_extractor.extract_hue(image)
    #feature_vector = append(feature_vector,color_extractor.calculate_histogram(hue, 20))

    # Then we add the shape_features
    #shape_features = shape_extractor.predictShape(hue)
    #feature_vector = append(feature_vector, shape_features)

    #TODO: extract symbol/icon features

    feature_vectors.append(feature_vector)

"""
clf = SVC(C=1.0, cache_size=3000, class_weight=None, kernel='linear', max_iter=-1, probability=True,
                  random_state=None, shrinking=False, tol=0.001, verbose=False)
"""
clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, intercept_scaling=1,
                         class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                         multi_class='ovr', verbose=0)
clf.fit(feature_vectors, results)

# Testing phase
prediction_object = Prediction()
for im in test_images:
    print("Predict: ", im)
    validation_feature_vector = color_extractor.extract_hog(im)

    # Extract the same color features as the training phase
    #hue = color_extractor.extract_hue(im)
    #validation_feature_vector = append(validation_feature_vector,color_extractor.calculate_histogram(hue, 20))

    # And the same shape features
    #shape_features = shape_extractor.predictShape(hue)
    #validation_feature_vector = append(validation_feature_vector, shape_features)
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