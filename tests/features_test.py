import scipy.io
import scipy.misc
import numpy

from pydro.features import *

def features_test():
    lenna_image = scipy.misc.imread('tests/lenna.png').astype(numpy.float32)
    lenna_features = ComputeFeatures(lenna_image, 8)

    correct_features = scipy.io.loadmat('tests/lenna_features.mat')

    assert (numpy.fabs(correct_features['features']-lenna_features) < 1e-6).all()
