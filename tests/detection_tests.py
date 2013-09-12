import scipy.io
import numpy

from pydro.detection import *
from pydro.features import *

def detection_test():
    data = scipy.io.loadmat('tests/detection_test_data.mat')

    detection = Filter(data['input'], data['filter'])

    padded = -2*numpy.ones(detection.shape)
    left_pad = (detection.shape[1]-data['detection'].shape[1])/2
    top_pad = (detection.shape[0]-data['detection'].shape[0])/2
    
    padded[top_pad:top_pad+data['detection'].shape[0], left_pad:left_pad+data['detection'].shape[1]] = data['detection']

    assert (numpy.fabs(padded - detection) < 1e-6).all()

def detection_input_test():
    data = scipy.io.loadmat('tests/detection_test_data.mat')
    image = scipy.misc.imread('tests/lenna.png').astype(numpy.float32)

    features = ComputeFeatures(image, 8)
    assert (numpy.fabs(data['input'] - features) < 1e-6).all()

   
