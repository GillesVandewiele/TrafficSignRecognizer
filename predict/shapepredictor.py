
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


def predictShape(contour):

        shape = ""
        mu = cv2.moments(contour,False)

        #Calculate first moment invariant
        I = (mu["mu20"]*mu["mu02"] - mu["mu11"]**2)/(mu["m00"]**4)

        #Measure ellipticity
        E = 0
        if I < 1/(16*math.pi**2):
            E = 16*(math.pi**2)*I
        else:
            E =  1/(16*(math.pi**2)*I)

        #Measure triangularity
        T = 0
        if I < 1/108:
            T = 108*I
        else:
            T =  1/(108*I)

        #Measure octagonality
        O = 0
        if I < 1/(15.932*math.pi**2):
            O = 15.932*(math.pi**2)*I
        else:
            O =  1/(15.932*(math.pi**2)*I)

        #Measure Rectangularity
        minRect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(minRect)
        area = cv2.contourArea(contour)
        areaMinRect = cv2.contourArea(box)
        R = area / areaMinRect

        # print("I: " + str(I))
        # print("E: " + str(E))
        # print("T: " + str(T))
        # print("O: " + str(O))
        # print("R: " + str(R))

        if E > T and E > O and E > R:
            return "Circle"
        elif T > E and T > O and T > R:
            return "Triangle"
        elif R > E and R > T and R > O:
            return "Rectangle"
        elif O > E and O > T and O > R:
            return "Diamond"

        return shape


def doPredictShapes(img):
        #concat = root + "\\" + file.title()
        #img = cv2.imread(concat,0)
        ret, thresh = cv2.threshold(img, 127, 255,0)
        imgg, contours, hierarchy = cv2.findContours(thresh,2,1)
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
        cnt = contours[index]

        # minRect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(minRect)
        # box = np.int0(box)

        #convert grayscale to color image
        # img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(img_rgb, cnt, -1, (255,0,0), 3)
        # cv2.drawContours(img_rgb, [box], -1, (255,0,0), 3)
        # cv2.imshow("window title", img_rgb)
        # cv2.waitKey()

        shape = predictShape(cnt)
        print("Shape image: " + shape)




path = 'C:\\Users\Tim Deweert\Documents\GitHub\TrafficSignRecognizer\\train\\blue_circles'
os.chdir(path);
print(os.getcwd())

train_images_dir = 'C:\\Users\Tim Deweert\Documents\GitHub\TrafficSignRecognizer\\train\\blue_circles'

"""
train_images = []
for root, subFolders, files in os.walk(train_images_dir):
    for file in files:
        train_images.append(os.path.join(root, file))
        #print(root + "   " + file.title())
        doPredictShapes(root,file.title())


"""
"""

root = "C:\Users\Tim Deweert\Documents\GitHub\TrafficSignRecognizer\\train\\blue_circles\\D10\\"
file = "00241_01783.png"
doPredictShapes(root,file.title())


"""





























# img1 = cv2.imread('circle3.jpg',0)
# img2 = cv2.imread('triangle1.png',0)
#
# ret, thresh = cv2.threshold(img1, 127, 255,0)
# ret, thresh2 = cv2.threshold(img2, 127, 255,0)
#
#
# img, contours, hierarchy = cv2.findContours(thresh,2,1)
# index = 0;
# max_area = cv2.contourArea(contours[0])
# for i in range(0,len(contours)):
#
#     # img_rgb = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
#     # cv2.drawContours(img_rgb, contours[i], -1, (255,0,0), 5)
#     # cv2.imshow("window title", img_rgb)
#     # cv2.waitKey()
#
#     area = cv2.contourArea(contours[i])
#     if area > max_area:
#         max_area = area
#         index = i
# cnt1 = contours[index]
#
#
#
# img, contours, hierarchy = cv2.findContours(thresh2,2,1)
# index = 0;
# max_area = cv2.contourArea(contours[0])
# for i in range(0,len(contours)):
#
#     # img_rgb = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
#     # cv2.drawContours(img_rgb, contours[i], -1, (255,0,0), 5)
#     # cv2.imshow("window title", img_rgb)
#     # cv2.waitKey()
#
#     area = cv2.contourArea(contours[i])
#     if area > max_area:
#         max_area = area
#         index = i
# cnt2 = contours[index]
#
#
# minRect = cv2.minAreaRect(cnt2)
# box = cv2.boxPoints(minRect)
# box = np.int0(box)
#
# #convert grayscale to color image
# img_rgb1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
# img_rgb2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
#
# cv2.drawContours(img_rgb1, cnt1, -1, (255,0,0), 3)
# cv2.imshow("window title", img_rgb1)
# cv2.waitKey()
#
# cv2.drawContours(img_rgb2, cnt2, -1, (255,0,0), 3)
# cv2.drawContours(img_rgb2, [box], -1, (255,0,0), 3)
# cv2.imshow("window title", img_rgb2)
# cv2.waitKey()
#
# # cv2.imshow('image',img)
# # cv2.waitKey(0)
#
# shape1 = predictShape(cnt1)
# shape2 = predictShape(cnt2)
#
# print("Shape 1: " + shape1)
# print("Shape 2: " + shape2)
#
# ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
# print(ret)

