from PIL import Image
import xlsxwriter

__author__ = 'Group16'

"""

    This class contains the code to:
        - Read a image as input
        - Write a csv file as output

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""

class FileParser:

    def __init__(self):
        pass

    def readImage(self, path):
        return Image.open(path)

    def writeCSV(self, path):
        workbook = xlsxwriter.Workbook(path)

"""
import os
im = Image.open(os.path.join(os.path.dirname(__file__), "../00062_04919.png"));
print(im.__dict__);
im.rotate(45).show();
"""