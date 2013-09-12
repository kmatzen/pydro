import numpy.linalg
import scipy.io
import scipy.misc
import numpy
import itertools

from pydro.features import *
from pydro.io import *

def features_test():
    lenna_image = scipy.misc.imread('tests/lenna.png').astype(numpy.float32)
    lenna_features = ComputeFeatures(lenna_image, 8)

    correct_features = scipy.io.loadmat('tests/lenna_features.mat')

    assert (numpy.fabs(correct_features['features']-lenna_features) < 1e-6).all()

def build_pyramid_test():
    image = scipy.misc.imread('tests/lenna.png')

    sbin = 8
    interval = 10

    pyramid = list(BuildPyramid(image, sbin, interval, True))

    for level in pyramid:
        assert math.floor(image.shape[0]*level.scale/sbin) - level.features.shape[0] <= 2
        assert math.floor(image.shape[1]*level.scale/sbin) - level.features.shape[1] <= 2

    pyramid.sort(key=lambda k: k.scale)

    sc = 2**(1.0/interval)
    scale = pyramid[0].scale
    for level in pyramid[1:]:
        new_scale = level.scale
        assert math.fabs(new_scale/scale - sc) < 1e-6
        scale = new_scale

def compare_pyramid_test():
    image = scipy.misc.imread('tests/000034.jpg')

    sbin = 8
    interval = 10

    pyramid = list(BuildPyramid(image, sbin, interval, False))
    pyramid.sort(key=lambda k: -k.scale)

    model = LoadModel('tests/example.dpm')
    correct = scipy.io.loadmat('tests/pyramid.mat')

    pyra = correct['pyra']

    assert (numpy.fabs(pyra[0][0][1].flatten() - numpy.array([l.scale for l in pyramid])) < 1e-6).all()

    for level, given in itertools.izip(pyramid, pyra[0][0][0]):
        given = given[0]
        if level.features.shape == given.shape:
            assert (level.features - given).sum()/level.features.size < 1e-2
