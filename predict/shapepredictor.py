
import os
import cv2
from PIL import Image
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from predict.predictor import Predictor

__author__ = 'Group16'

"""

    Interface for the Predictor class

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

class ShapePredictor():

    @staticmethod
    def predictShape(hue):
            img = (np.rint(asarray(hue)*255)).astype(np.uint8)
            ret, thresh = cv2.threshold(img, 127, 255,0)
            imgg, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,1)
            if len(contours) == 0:
                return [0,0,0,0]
            index = 0;
            max_area = cv2.contourArea(contours[0])
            for i in range(0,len(contours)):
                area = cv2.contourArea(contours[i])
                if area > max_area:
                    max_area = area
                    index = i
            contour = contours[index]

            mu = cv2.moments(contour,False)

            #The zero order central moment is zero so no shape can be predicted
            if(mu["m00"] == 0):
                return [0,0,0,0]

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

            return[E,T,O,R]


