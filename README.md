# MLGroup16
The repository for group 16 for the course Machine Learning. Detecting traffic signs on pictures

# Used Libraries

## XLSXWriter

## numpy, pylab, skimage, sklearn, scipy, 


## OpenCV
For shape matching, you must install OpenCV. I used this to get it working:

Gohlke maintains Windows binaries for many Python packages, including OpenCV 3.0 with Python 3.x bindings! See here:

http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

To install, just download the 64-bit or 32-bit .whl file appropriate for your system, then run pip install [filename]. 
Then the instruction import cv2 should work in your Python 3.x interpreter.

## Mahotas

For calculating the zernike moments, we used a library called Mahotas. You can download
a build of it from the website:

http://www.lfd.uci.edu/~gohlke/pythonlibs/#mahotas

To install, just download the 64-bit or 32-bit .whl file appropriate for your system, then run pip install [filename]. 
Then the instruction import cv2 should work in your Python 3.x interpreter.

## Lasagne

Lasagne is a lightweight library to build and train neural networks in Theano.
How it can be installed is explained in their docs:

http://lasagne.readthedocs.org/en/latest/user/installation.html


# IDE: Pycharm (JetBrains)


# How to run

All variables can be declared in main.py and feature extractors can be selected.

`tsr.make_submission(train_images_path=train_images_dir, test_images_path=test_images_dir, output_file_path="test.xlsx", feature_extractors=feature_extractors, size=64)`

`print(tsr.local_test(train_images_path=train_images_dir, feature_extractors=feature_extractors, k=2, nr_data_augments=1, size=64))`

## Bug

We also have a bug for making a submisson for which we didn't have time to fix. Depending on the model, line 38 should be commentend or not in fileparser.py.  
For the linear model and random forest, the line should be included:  
	`sorted_signs = sorted(predictionObject.TRAFFIC_SIGNS)`  
For the neural network the line should be commented:  
  	`#sorted_signs = sorted(predictionObject.TRAFFIC_SIGNS)`  
