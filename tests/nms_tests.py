from pydro.io import *
from pydro.features import *
from pydro.detection import *

import itertools
import scipy.misc

def nms_small_test():
    model = LoadModel('tests/example.dpm')
    model.start.rules = model.start.rules[:1]
    model.start.rules[0].rhs = model.start.rules[0].rhs[1:2]
    model.start.rules[0].anchor = model.start.rules[0].anchor[1:2]

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model=model)

    filtered_model = model.Filter(pyramid)

    detections = filtered_model.Parse(-2)

    nms = NMS (detections, 2)

def nms_test():
    model = LoadModel('tests/example.dpm')

    image = scipy.misc.imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model=model)

    filtered_model = model.Filter(pyramid)

    detections = filtered_model.Parse(-2)

    nms = NMS (detections, 2)
