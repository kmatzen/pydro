import scipy.io
import numpy

from pydro.detection import *
from pydro.features import *
from pydro.io import *

from pydro.io import _type_handler

def detection_test():
    data = scipy.io.loadmat('tests/detection_test_data.mat')

    detection = FilterImage(data['input'], data['filter'])

    left_pad = (detection.shape[1]-data['detection'].shape[1])/2
    top_pad = (detection.shape[0]-data['detection'].shape[0])/2
    
    assert (numpy.fabs(data['detection'] - detection[top_pad:top_pad+data['detection'].shape[0], left_pad:left_pad+data['detection'].shape[1]]) < 1e-6).all()

def detection_input_test():
    data = scipy.io.loadmat('tests/detection_test_data.mat')
    image = scipy.misc.imread('tests/lenna.png').astype(numpy.float32)

    features = ComputeFeatures(image, 8, 0, 0)
    assert (numpy.fabs(data['input'] - features) < 1e-6).all()

def filter_model_test():
    model = LoadModel('tests/example.dpm')

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model.sbin, model.interval, model.features.extra_octave, model.maxsize[1]+1, model.maxsize[0]+1)

    filtered_model = model.Filter(pyramid)

    correct = scipy.io.loadmat('tests/scores.mat')

    for i in xrange(len(filtered_model.filtered_start.score)):
        mine = filtered_model.filtered_start.score[i].score
        given = correct['score'][0,i]

        if not isinstance(mine, numpy.ndarray):
            assert mine == -numpy.inf
            assert (given == -numpy.inf).all()
            continue
        Iy1, Ix1 = numpy.where(mine == -numpy.inf)
        Iy2, Ix2 = numpy.where(given == -numpy.inf)


        assert (Iy1 == Iy2).all()
        assert (Ix1 == Ix2).all()

        diff = given - mine
        diff = diff[numpy.logical_not(numpy.isnan(diff))]

        if diff.size > 0:
            print(filtered_model.filtered_start.score[i].scale, numpy.fabs(diff).max())
            if filtered_model.filtered_start.score[i].scale > 1:
                assert numpy.fabs(diff).max() < 2e-1
            else:
                assert numpy.fabs(diff).max() < 2e-1

def filter_model_small_test():
    model = LoadModel('tests/example.dpm')
    model.start.rules = model.start.rules[:1]
    model.start.rules[0].rhs = model.start.rules[0].rhs[1:2]
    model.start.rules[0].anchor = model.start.rules[0].anchor[1:2]

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model.sbin, model.interval, model.features.extra_octave, model.maxsize[1]+1, model.maxsize[0]+1)

    filtered_model = model.Filter(pyramid)

    correct = scipy.io.loadmat('tests/scored.mat')

    for i in xrange(len(filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].filtered_rhs[0].score)):
        mine = filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].filtered_rhs[0].score[i].score
        given = correct['model_scored'][0,0][5][0,1][2][0,i]

        if filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].filtered_rhs[0].score[i].scale > 1:
            assert numpy.fabs(given - mine).max() < 1e-1
        else:
            assert numpy.fabs(given - mine).max() < 1e-2

    for i in xrange(len(filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].score)):
        mine = filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].score[i].score
        given = correct['model_scored'][0,0][4][0,2][0,0][10][0,i]

        if filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].score[i].scale > 1:
            assert numpy.fabs(given - mine).max() < 1e-1
        else:
            assert numpy.fabs(given - mine).max() < 1e-2

    for i in xrange(len(filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].score)):
        mine = filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].score[i].score
        given = correct['model_scored'][0,0][5][0,2][2][0,i]

        if filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].score[i].scale > 1:
            assert numpy.fabs(given - mine).max() < 1e-1
        else:
            assert numpy.fabs(given - mine).max() < 1e-2

    for i in xrange(len(filtered_model.filtered_start.filtered_rules[0].score)):
        mine = filtered_model.filtered_start.filtered_rules[0].score[i].score
        given = correct['model_scored'][0,0][4][0,0][0,0][10][0,i]

        if not isinstance(mine, numpy.ndarray):
            assert mine == -numpy.inf
            assert (given == -numpy.inf).all()
            continue

        Iy1, Ix1 = numpy.where(mine == -numpy.inf)
        Iy2, Ix2 = numpy.where(given == -numpy.inf)

        assert (Iy1 == Iy2).all()
        assert (Ix1 == Ix2).all()

        diff = given - mine
        diff = diff[numpy.logical_not(numpy.isnan(diff))]

        if diff.size > 0:
            print(numpy.fabs(diff).max())
            if filtered_model.filtered_start.filtered_rules[0].score[i].scale > 1:
                assert numpy.fabs(diff).max() < 1e-1
            else:
                assert numpy.fabs(diff).max() < 2e-2

    for i in xrange(len(filtered_model.filtered_start.score)):
        mine = filtered_model.filtered_start.score[i].score
        given = correct['model_scored'][0,0][5][0,0][2][0,i]

        if not isinstance(mine, numpy.ndarray):
            assert mine == -numpy.inf
            assert (given == -numpy.inf).all()
            continue

        Iy1, Ix1 = numpy.where(mine == -numpy.inf)
        Iy2, Ix2 = numpy.where(given == -numpy.inf)

        assert (Iy1 == Iy2).all()
        assert (Ix1 == Ix2).all()

        diff = given - mine
        diff = diff[numpy.logical_not(numpy.isnan(diff))]

        if diff.size > 0:
            print(numpy.fabs(diff).max())
            if filtered_model.filtered_start.score[i].scale > 1:
                assert numpy.fabs(diff).max() < 1e-1
            else:
                assert numpy.fabs(diff).max() < 2e-2


def deformation_test():
    data = scipy.io.loadmat('tests/deformation_example.mat')

    values = numpy.array(data['values'], dtype=numpy.float32, order='C')
    deformed, Ix, Iy = DeformationCost(values, 1, 1, 1, 1, 4)

    assert (numpy.fabs(deformed - data['A']) < 1e-6).all()
    assert (Ix + 1 == data['Ix']).all()
    assert (Iy + 1 == data['Iy']).all()

def deformation2_test():
    data = scipy.io.loadmat('tests/deformation_example.mat')

    values = numpy.array(data['values'], dtype=numpy.float32, order='C')
    deformed, Ix, Iy = DeformationCost(values, 0.1, 0.2, 0.1, 0.02, 4)

    assert (numpy.fabs(deformed - data['A2']) < 1e-6).all()
    assert (Ix + 1 == data['Ix2']).all()
    assert (Iy + 1 == data['Iy2']).all()
