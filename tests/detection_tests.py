import scipy.io
import numpy

from pydro.detection import *
from pydro.features import *
from pydro.io import *

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
    pyramid = BuildPyramid(image, model.sbin, model.interval, model.features.extra_octave)

    filtered_model = model.Filter(pyramid)

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
