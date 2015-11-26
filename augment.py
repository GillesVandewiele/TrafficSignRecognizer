import os
import random
from scipy import ndimage
import cv2
import numpy as np

#################
# 2D Convolution
#################
def convolution_2d(img):
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img, -1, kernel)

####################
# Gaussian Blurring
####################
def gauss_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

####################
# Rotating
####################
def rotate_img_rand(img):
    angle = random.randint(-5, 5)
    return ndimage.rotate(img, angle)

train_images_dir = os.path.join(os.path.dirname(__file__), "train")
for root, subFolders, files in os.walk(train_images_dir):
    for file in files:
        print("Transforming ", os.path.join(root, file))
        img = cv2.imread(os.path.join(root, file))
        cv2.imwrite(os.path.join(root, file[:-4])+"_conv.png", convolution_2d(img))
        cv2.imwrite(os.path.join(root, file[:-4])+"_blur.png", gauss_blur(img))
        cv2.imwrite(os.path.join(root, file[:-4])+"_rot.png", rotate_img_rand(img))