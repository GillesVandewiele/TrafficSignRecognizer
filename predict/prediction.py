import math

__author__ = 'Group16'

"""

    This is the object for prediction data

    Written by Group 16: Tim Deweert, Karsten Goossens & Gilles Vandewiele
    Commissioned by UGent, course Machine Learning

"""


class Prediction(object):
    """
        :var predictions: an array with for each picture a dictionary (key="sign string"; value = "probability")
                          of possibilities for each kind of traffic sign
    """

    TRAFFIC_SIGNS = [
                     # Blue Circles
                     "D1a",     # rechtdoor rijden
                     "D1b",     # naar rechts/links
                     "D1e",     # rechtdoor en naar rechts/links
                     "D5",      # Rotonde
                     "D7",      # Fietsers
                     "D9",      # Voetgangers en fietsers naast elkaar (voetgangers kunnen links en rechts staan)
                     "D10",     # Voetgangers en fietsers onder elkaar

                     # Diamonds
                     "B9",      # Voorrangsbaan
                     "B11",     # Geen voorrangsbaan

                     # Forbidden
                     "C1",      # Verboden toegang

                     # Other
                     "begin",   # Geldt vanaf hier (wit bord met zwarte streep naar boven)
                     "C37",     # Chinees wit bord met zwarte streep
                     "e0c",     # Geldt vanaf hier voor x meter
                     "end",     # Eindigt hier
                     "F1",      # Gemeentebord (wit) zonder tekening
                     "F1a_h",   # Gemeentebord met tekening
                     "F3a_h",   # Beide gemeenteborden met rode streep
                     "F4b",     # Einde zone 30
                     "F13",     # Aanwijzingen voorvoegvakken
                     "F21",     # Beide richtingen
                     "F23A",    # Nationale wegen (N485)
                     "F25",     # Richtingaanwijzers bij rotonde
                     "F27",     # Richtingaanwijzers met nationele wegen bij
                     "F29",     # Richtingaanwijzers met (mogelijks) kilometers
                     "F31",     # Aanwijzers autostrades (E40)
                     "F33_34",  # Lokale richtingaanwijzers (naar bowling & café's)
                     "F35",     # Toeristische richtingaanwijzers (bruin, bv naar voetbalploeg/bezienswaardigheid)
                     "F41",     # Richtingaanwijzer tijdens werken (rood)
                     "F43",     # Geel gemeentebord (bebouwde kom)
                     "Handic",  # Tim zijn parkeerplaats
                     "lang",    # Wit bord met pijl naar boven en beneden
                     "m",       # Uitgezonderd fietsers en bromfietsers

                     # Rectangles down
                     "F12a",    # Speelzone
                     "F12b",    # Geen speelzone

                     # Rectangles up
                     "B21",     # Tegenliggers hebben geen voorrang
                     "E9a",     # Parking
                     "E9a_miva",# Gilles zijn parking
                     "E9b",     # Parking voor auto's
                     "E9cd",    # Parking vrachtwagens
                     "E9e",     # Parking met hefboom
                     "F45",     # Doodlopende straat
                     "F47",     # Geen werken meer
                     "F59",     # Parking naar rechts
                     "X",       # Zone betalende parking

                     # Red Blue Circles
                     "E1",      # Verboden te parkeren
                     "E3",      # Verboden te parkeren aan beide kanten
                     "E5",      # Verboden te parkeren van 1 tot 15
                     "E7",      # Verboden te parkeren van 16 tot 31

                     # Red circles
                     "B19",     # Voorrang geven aan tegenligger
                     "C3",      # Verboden
                     "C11",     # Verboden voor fietsers
                     "C21",     # Verboden voor vrachtwagens over 3.5t
                     "C23",     # Verboden voor vrachtwagens
                     "C29",     # Verboden voor voertuigen hoger dan 3m20
                     "C31",     # Verboden rechtdoor en links/rechts
                     "C35",     # Verboden in te halen
                     "C43",     # Snelheidslimiet
                     "F4a",     # Zone 30

                     # Reversed Triangles
                     "B1",      # Voorrang van rechts
                     "B3",      # Voorrang van rechts binnen x meter
                     "B7",      # Voorrang van rechts + STOP binnen x meter

                     # Squares
                     "F19",     # Pijl rechtdoor
                     "F49",     # Voetgangers oversteekplaats
                     "F50",     # Fietsers oversteekplaats
                     "F87",     # Snelheidsheuvel

                     # Stop
                     "B5",      # Stop bord

                     # Triangles
                     "A1AB",    # Scherpe bocht naar links/rechts
                     "A1CD",    # S-bocht
                     "A7A",     # Versmalde doorgang
                     "A7B",     # Versmalling doorgang van een kant
                     "A13",     # Heuvels
                     "A14",     # Snelheidsdrempel
                     "A15",     # Gladdige baan
                     "A23",     # School
                     "A23_yellow", # Geel schoolbord
                     "A25",     # Opgelet fietsers
                     "A29",     # Opgelet koeien
                     "A31",     # Opgelet werken
                     "A51",     # Uitroepteken
                     "B15A",    # Voorrangsbaan van links en rechts&
                     "B17"      # Voorrang geven aan links en rechts
                     ]

    def __init__(self, predictions=None):
        if not predictions:
            predictions = []
        self.predictions = predictions

    def addPrediction(self, prediction, typechecking=False):
        """
            Add a prediction to the predictions list
            :param prediction: should be a dict of a constant length (number of different traffic signs)
            :param typechecking: boolean that can be put on True for debugging purposes (slows down object creation)
        """
        if typechecking:
            if len(prediction) != len(Prediction.TRAFFIC_SIGNS):
                raise PredictionException("The prediction dict must have a length of 81", prediction)
            if not all(key in Prediction.TRAFFIC_SIGNS for key in prediction):
                raise PredictionException("Mismatching keys", prediction)

        self.predictions.append(prediction)


    def addMultiplePredictions(self, newPredictions):
        for prediction in newPredictions:
            self.addPrediction(prediction)

    def evaluate(self, results):
        """
            Evaluate the logloss score
            :param results: array with the same length as :var predictions with the corresponding result as a string
            :return: the logloss score
        """
        logloss = 0
        counter = 0
        results_indices = sorted(set(results))
        for prediction in self.predictions:
            p = max(min(prediction[results_indices.index(results[counter])], 1-pow(10, -15)), pow(10, -15))
            logloss += math.log(p)
            counter += 1

        logloss /= -len(self.predictions)

        return logloss

    def evaluate_binary(self, results):
        """
            Evaluate the logloss score
            :param results: array with the same length as :var predictions with the corresponding result as a string
            :return: the logloss score
        """
        correct = 0
        counter = 0
        for prediction in self.predictions:
            if prediction == results[counter]:
                correct += 1
            counter += 1


        return correct/counter


class PredictionException(Exception):
    def __init__(self, value, prediction):
        self.value = value
        self.prediction = prediction
    def __str__(self):
        return repr(self.value) + self.prediction



