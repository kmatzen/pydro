from pydro.io import *
from pydro.train import *
from pydro.features import *

import itertools

import scipy.misc

def train_test():
    model = LoadModel('tests/example.dpm')

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model=model)

    filtered_model = model.Filter(pyramid)

    detections = [d for i,d in itertools.izip(xrange(1), filtered_model.Parse(-1))]

    feature_vector = [BuildFeatureVector(model, d, pyramid) for d in detections]

