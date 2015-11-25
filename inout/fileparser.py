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
    def write_CSV(path, predictionObject):
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet("submission")

        worksheet.write(0,0, "id")

        print(predictionObject)

        # Write out first row (all traffic signs)
        signCounter = 0
        for trafficSign in sorted(predictionObject.TRAFFIC_SIGNS):
            worksheet.write(0, signCounter+1, trafficSign)
            signCounter+=1


        sorted_signs = sorted(predictionObject.TRAFFIC_SIGNS)
        predictionCounter = 0
        for prediction in predictionObject.predictions:
            # Write out prediction id and afterwards 81 probabilities
            worksheet.write(predictionCounter+1, 0, predictionCounter+1)
            signCounter = 0
            for trafficSign in prediction:
                # The output must be written out sorted, so first get the string of the current index (e.g. 0 could be 'D10')
                # Then look for the corresponding column to write by matching the index in the sorted list (+1 for the id column)
                worksheet.write(predictionCounter+1, sorted_signs.index(predictionObject.TRAFFIC_SIGNS[signCounter])+1, trafficSign)
                signCounter += 1

            predictionCounter += 1

        workbook.close()