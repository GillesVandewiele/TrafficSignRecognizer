# import the necessary packages
import mahotas

__author__ = 'Group16'


class ZernikeMoments:

    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius)