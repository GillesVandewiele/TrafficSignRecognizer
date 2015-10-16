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

    def writeCSV(self, path, predictionObject):
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet("submission")

        worksheet.write(1,1, "id")

        # Write out first row (all traffic signs)
        signCounter = 1
        for trafficSign in predictionObject.TRAFFIC_SIGNS:
            workbook.write(1, signCounter+1, trafficSign)
            signCounter+=1

        predictionCounter = 1
        for prediction in predictionObject.predictions:
            # Write out prediction id and afterwards 81 probabilities
            workbook.write(predictionCounter+1, 1, predictionCounter)

            signCounter = 1
            for trafficSign in predictionObject.TRAFFIC_SIGNS:
                workbook.write(predictionCounter+1, signCounter+1, prediction[trafficSign])

        workbook.close()