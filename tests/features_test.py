import numpy.linalg
import scipy.io
import scipy.misc
import numpy
import itertools

from pydro.features import *
from pydro.io import *

def resize_test():
    image = scipy.misc.imread('tests/lenna.png').astype(numpy.float32)

    image_resized = ResizeImage(image, image.shape[0]/2, image.shape[1]/2)
    

def features_test():
    lenna_image = scipy.misc.imread('tests/lenna.png').astype(numpy.float32)
    lenna_features = ComputeFeatures(lenna_image, 8, 0, 0)

    correct_features = scipy.io.loadmat('tests/lenna_features.mat')

    assert (numpy.fabs(correct_features['features']-lenna_features) < 1e-6).all()

def build_pyramid_test():
    image = scipy.misc.imread('tests/lenna.png')

    sbin = 8
    interval = 10

    pyramid = BuildPyramid(image, sbin, interval, True, 16, 7)

    for level in pyramid:

        assert math.floor(image.shape[0]*level.scale/sbin) - level.features.shape[0] - 30 <= 2
        assert math.floor(image.shape[1]*level.scale/sbin) - level.features.shape[1] - 6 <= 2
        assert level.features.shape[2] == 32

        assert (numpy.fabs(level.features[:7,:,:-1]) < 1e-6).all()
        assert (numpy.fabs(level.features[:7,:,-1] - 1) < 1e-6).all()

        assert (numpy.fabs(level.features[-7:,:,:-1]) < 1e-6).all()
        assert (numpy.fabs(level.features[-7:,:,-1] - 1) < 1e-6).all()

        assert (numpy.fabs(level.features[:,:16,:-1]) < 1e-6).all()
        assert (numpy.fabs(level.features[:,:16,-1] - 1) < 1e-6).all()

        assert (numpy.fabs(level.features[:,-16:,:-1]) < 1e-6).all()
        assert (numpy.fabs(level.features[:,-16:,-1] - 1) < 1e-6).all()

    sc = 2**(1.0/interval)
    scale = pyramid[0].scale
    for level in pyramid[1:]:
        new_scale = level.scale
        assert math.fabs(new_scale/scale - 1/sc) < 1e-6
        scale = new_scale

def compare_pyramid_test():
    image = scipy.misc.imread('tests/000034.jpg')

    model = LoadModel('tests/example.dpm')
    pyramid = BuildPyramid(image, model.features.sbin, model.interval, False, 16, 7)

    correct = scipy.io.loadmat('tests/pyramid.mat')

    pyra = correct['pyra']

    assert (numpy.fabs(pyra[0][0][1].flatten() - numpy.array([l.scale for l in pyramid])) < 1e-6).all()

    for level, given in itertools.izip(pyramid, pyra[0][0][0]):
        given = given[0]
        assert level.features.shape == given.shape
        diff = level.features/given
        diff = diff[numpy.logical_not(numpy.isnan(diff))]
        diff = diff[diff != numpy.inf]
        diff = diff[diff != -numpy.inf]
        if level.scale > 1:
            assert numpy.fabs(numpy.median(numpy.fabs(diff))-1) < 1e-2
        else:
            print(numpy.median(numpy.fabs(diff)))
            assert numpy.fabs(numpy.median(numpy.fabs(diff))-1) < 1e-1

def resize_test():
    correct = scipy.io.loadmat('tests/resize_test.mat')

    im = correct['im']
    im_small_correct = correct['im_small']

    im = numpy.array(im, dtype=numpy.float32, order='C')
    im_small_mine = ResizeImage(im.astype(numpy.float32), im_small_correct.shape[0], im_small_correct.shape[1])

    assert numpy.fabs(im_small_correct - im_small_mine).max() < 1e-2
