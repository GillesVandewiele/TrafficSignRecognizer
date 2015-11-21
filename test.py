# http://www.maths.lth.se/matematiklth/personal/solem/downloads/vlfeat.py
from PIL import Image
import os
from math import sqrt
from pylab import *
from scipy.cluster import vq
from vlfeat_wrapper import process_image, read_features_from_file, plot_features, match_twosided, plot_matches, match

PRE_ALLOCATION_BUFFER = 1000  # for sift
K_THRESH = 0.1

def extractSift(input_files):
    all_features_dict = {}
    for i, fname in enumerate(input_files):
        print("calculating sift features for", fname)
        process_image(fname, 'tmp.sift')
        locs, descriptors = read_features_from_file('tmp.sift')
        print(len(descriptors), len(descriptors[0]))
        all_features_dict[fname] = descriptors
    return all_features_dict

def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    print(array)
    array = resize(array, (pivot, 128))
    return array

def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words

def get_images_from_directory(directory):
    # Get all files from a directory
    images = []
    for root, subFolders, files in os.walk(directory):
        for file in files:
            images.append(os.path.join(root, file))

    return images

train_images_dir = os.path.join(os.path.dirname(__file__), "train")  # Train image directory
train_images = get_images_from_directory(train_images_dir)
train_images = train_images[500:1000]

features = extractSift(train_images)
all_features_array = dict2numpy(features)
print(all_features_array.shape)
nfeatures = all_features_array.shape[0]
nclusters = int(sqrt(nfeatures))
codebook, distortion = vq.kmeans(all_features_array,
                                         nclusters,
                                         thresh=K_THRESH)

print(codebook)
print(distortion)

all_word_histgrams = {}
for imagefname in features:
    word_histgram = computeHistograms(codebook, features[imagefname])
    all_word_histgrams[imagefname] = word_histgram

print(all_word_histgrams)

"""
process_image('00661_02761.png', 'tmp.sift')
l,d = read_features_from_file('tmp.sift')

print(len(l), len(d))

im = Image.open('00661_02761.png')
im = array(im)
figure()
plot_features(im,l,True)
show()

process_image('00630_01624.png','tmp2.sift')
l2,d2 = read_features_from_file('tmp2.sift')
im2 = array(Image.open('00630_01624.png'))

m = match(d,d2)

print(m)
print(len(l), len(d))

process_image('00023_10428.png','tmp2.sift')
l2,d2 = read_features_from_file('tmp2.sift')
im2 = array(Image.open('00023_10428.png'))

m = match(d,d2)

print(m)
print(len(l), len(d))
"""


