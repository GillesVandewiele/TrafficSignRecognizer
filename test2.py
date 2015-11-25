import os
import cv2
import sys
import time
import numpy as np
import random
from pylab import *
from scipy.stats import rv_discrete
from skimage import color
from skimage.transform import resize
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from inout.fileparser import FileParser
from predict.benchmark import BenchmarkPredictor
from predict.colorfeatureextractor import ColorFeatureExtractor
from predict.hogfeatureextractor import HogFeatureExtractor
from predict.prediction import Prediction
from predict.shapefeatureextractor import ShapeFeatureExtractor
from predict.symbolfeatureextractor import SymbolFeatureExtractor
import lasagne
import theano
import theano.tensor as T



def get_results(train_images_dir):
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
 
def get_images_from_directory(directory):
    # Get all files from a directory
    images = []
    for root, subFolders, files in os.walk(directory):
        for file in files:
            images.append(os.path.join(root, file))
 
    return images
 
 
def preprocess_image(image, extractor):
    image_array = cv2.imread(image)
    im = resize(image_array, (64, 64, 3))
    return extractor.extract_feature_vector(im)
 
 
 
def build_mlp(input_var=None):
 
    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 576),
                                     input_var=input_var)
    # Another 800-unit layer:
    l_hid = lasagne.layers.DenseLayer(
            l_in, num_units=100,
            nonlinearity=None)
 
    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid, num_units=81,
            nonlinearity=lasagne.nonlinearities.softmax)
 
    return l_out
 
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
 
def main(num_epochs=2000):
    # Load the dataset
    print("Loading data...")
    # X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
 
    train_images_dir = os.path.join(os.path.dirname(__file__), "train")
    test_images_dir = os.path.join(os.path.dirname(__file__), "test")
    train_images = get_images_from_directory(train_images_dir)
    test_images = get_images_from_directory(test_images_dir)
    train_results = get_results(train_images_dir)
    #test_results = get_results(test_images_dir)
    cfe = HogFeatureExtractor(8)
   
    all_train_images = []
    for image in train_images:
        all_train_images.append(np.asarray([preprocess_image(image,cfe)]))

    all_test_images = []
    for image in test_images:
        all_test_images.append(np.asarray([preprocess_image(image, cfe)]))

    X_test = np.asarray(all_test_images)
    #y_test = np.asarray(test_results)

    X_train = np.asarray(all_train_images)
    y_train = np.asarray(train_results)

    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_mlp(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    pred_fn = theano.function([input_var], test_prediction)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:

    prediction_object = Prediction()

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()
        train_err = train_fn(X_train, y_train)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss: ", train_err)

    # After training, we compute and print the test error:
    for index in range(len(X_test)):
        #print(test_images[index])
        #print(pred_fn([X_test[index]])[0])
        prediction_object.addPrediction(pred_fn([X_test[index]])[0])

    FileParser.write_CSV("submission.xlsx",prediction_object)
    #err, acc = val_fn(X_test, y_test)

    print("Final results:")
    #print("  test loss: ", err)
    #print("  test accuracy: ", acc * 100)
 
        # Optionally, you could now dump the network weights to a file like this:
        # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)
 
main()