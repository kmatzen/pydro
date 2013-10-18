from pydro.io import *
from pydro.train import *
from pydro.features import *
from pydro.core import *

import itertools

import scipy.misc

def train_test():
    model = LoadModel('tests/example.dpm')

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model=model)

    filtered_model = model.Filter(pyramid)

    detections = [d for i,d in itertools.izip(xrange(1), filtered_model.Parse(-1))]
    assert detections[0].loss is None

    feature_vector = [BuildFeatureVector(filtered_model, d) for d in detections]

def loss_adjustment_test():
    model = LoadModel('tests/example.dpm')

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model=model)

    def loss_adjustment (score, block):
        return [Score(scale=s.scale, score=s.score+1) for s in score]

    filtered_model = model.Filter(pyramid, loss_adjustment=loss_adjustment)

    detections = [d for i,d in itertools.izip(xrange(1), filtered_model.Parse(-1))]
    assert math.fabs(detections[0].loss - 9) < 1e-5
