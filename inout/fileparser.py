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

    @staticmethod
    def read_image(path):
        return Image.open(path)

    @staticmethod
    def write_CSV(path, predictionObject):
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet("submission")

        worksheet.write(0,0, "id")

        # Write out first row (all traffic signs)
        signCounter = 0
        for trafficSign in predictionObject.TRAFFIC_SIGNS:
            worksheet.write(1, signCounter+1, trafficSign)
            signCounter+=1

        predictionCounter = 0
        for prediction in predictionObject.predictions:
            # Write out prediction id and afterwards 81 probabilities
            worksheet.write(predictionCounter+1, 1, predictionCounter)

            signCounter = 0
            for trafficSign in predictionObject.TRAFFIC_SIGNS:
                worksheet.write(predictionCounter+1, signCounter+1, prediction[trafficSign])
                signCounter += 1

            predictionCounter += 1

        workbook.close()