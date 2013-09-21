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

    features = ComputeFeatures(image, 8)
    assert (numpy.fabs(data['input'] - features) < 1e-6).all()

def filter_model_test():
    model = LoadModel('tests/example.dpm')

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model.sbin, model.interval, model.features.extra_octave, model.maxsize[1], model.maxsize[0])

    filtered_model = model.Filter(pyramid)

    correct = scipy.io.loadmat('tests/scores.mat')

    for i in xrange(len(filtered_model.filtered_start.score)):
        mine = filtered_model.filtered_start.score[i].score
        given = correct['score'][0,i]

        if mine.shape == given.shape:
            diff = given/mine
            diff[numpy.isnan(diff)] = 0
            diff[numpy.where(diff == numpy.inf)] = 0
            diff[numpy.where(diff == -numpy.inf)] = 0
            diff = diff[numpy.where(diff != 0)]
            assert (numpy.fabs(numpy.fabs(diff).mean() - 1) < 1e-1).all()

def filter_model_small_test():
    model = LoadModel('tests/example.dpm')
    model.start.rules = model.start.rules[:1]
    model.start.rules[0].rhs = model.start.rules[0].rhs[1:2]
    model.start.rules[0].anchor = model.start.rules[0].anchor[1:2]

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model.sbin, model.interval, model.features.extra_octave, model.maxsize[1], model.maxsize[0])

    filtered_model = model.Filter(pyramid)

    correct = scipy.io.loadmat('tests/scored.mat')

    for i in xrange(len(filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].filtered_rhs[0].score)):
        mine = filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].filtered_rhs[0].score[i].score
        given = correct['model_scored'][0,0][5][0,1][2][0,i]

        if mine.shape == given.shape:
            assert (numpy.fabs(given - mine) < 1e-1).all()

    for i in xrange(len(filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].score)):
        mine = filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].filtered_rules[0].score[i].score
        given = correct['model_scored'][0,0][4][0,2][0,0][10][0,i]

        if mine.shape == given.shape:
            assert (numpy.fabs(given - mine) < 1e-1).all()

    for i in xrange(len(filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].score)):
        mine = filtered_model.filtered_start.filtered_rules[0].filtered_rhs[0].score[i].score
        given = correct['model_scored'][0,0][5][0,2][2][0,i]

        if mine.shape == given.shape:
            assert (numpy.fabs(given - mine) < 1e-1).all()

    for i in xrange(len(filtered_model.filtered_start.filtered_rules[0].score)):
        mine = filtered_model.filtered_start.filtered_rules[0].score[i].score
        given = correct['model_scored'][0,0][4][0,0][0,0][10][0,i]

        if mine.shape == given.shape:
            if numpy.where(mine == -numpy.inf)[0].shape == numpy.where(given == -numpy.inf)[0].shape:
                assert (numpy.fabs(given[numpy.where(given != -numpy.inf)] - mine[numpy.where(mine != -numpy.inf)]) < 1e-1).all()

def deformation_test():
    data = scipy.io.loadmat('tests/deformation_example.mat')

    values = numpy.array(data['values'], dtype=numpy.float32, order='C')
    deformed = DeformationCost(values, 1, 1, 1, 1, 4)

    assert (numpy.fabs(deformed - data['A']) < 1e-6).all()

def deformation_test():
    data = scipy.io.loadmat('tests/deformation_example.mat')

    values = numpy.array(data['values'], dtype=numpy.float32, order='C')
    deformed = DeformationCost(values, 0.1, 0.2, 0.1, 0.02, 4)

    assert (numpy.fabs(deformed - data['A2']) < 1e-6).all()
