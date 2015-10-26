
import os
import cv2
from PIL import Image
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image

__author__ = 'Group16'

"""

    Interface for the Predictor class

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""



"""
    Shapes:
        - circle
        - rectangle
        - diamond
        - square
"""

path = 'C:\\Users\Tim Deweert\Documents\GitHub\TrafficSignRecognizer'
os.chdir(path);
print(os.getcwd())


# read image to array
image = array(Image.open('test3.jpg').convert('L'))

img1 = cv2.imread('test2.jpg',0)
img2 = cv2.imread('testC2.jpg',0)

ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)


img, contours, hierarchy = cv2.findContours(thresh,2,1)
index = 0;
max_area = cv2.contourArea(contours[0])
for i in range(0,len(contours)):

    # img_rgb = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(img_rgb, contours[i], -1, (255,0,0), 5)
    # cv2.imshow("window title", img_rgb)
    # cv2.waitKey()

    area = cv2.contourArea(contours[i])
    if area > max_area:
        max_area = area
        index = i
cnt1 = contours[index]



img, contours, hierarchy = cv2.findContours(thresh2,2,1)
index = 0;
max_area = cv2.contourArea(contours[0])
for i in range(0,len(contours)):

    img_rgb = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_rgb, contours[i], -1, (255,0,0), 5)
    cv2.imshow("window title", img_rgb)
    cv2.waitKey()

    area = cv2.contourArea(contours[i])
    if area > max_area:
        max_area = area
        index = i
cnt2 = contours[index]



#convert grayscale to color image
img_rgb1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
img_rgb2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)

cv2.drawContours(img_rgb1, cnt1, -1, (255,0,0), 5)
cv2.imshow("window title", img_rgb1)
cv2.waitKey()

cv2.drawContours(img_rgb2, cnt2, -1, (255,0,0), 5)
cv2.imshow("window title", img_rgb2)
cv2.waitKey()

# cv2.imshow('image',img)
# cv2.waitKey(0)

ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print(ret)
