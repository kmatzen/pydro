from pydro.detection import *
from pydro.io import *
from pydro.features import *
from pydro.vis import *

import itertools
from scipy.misc import imshow, imread

def imshow(_):
    pass

def vis_small_test():
    model = LoadModel('tests/example.dpm')
    model.start.rules = model.start.rules[:1]
    model.start.rules[0].rhs = model.start.rules[0].rhs[1:2]
    model.start.rules[0].anchor = model.start.rules[0].anchor[1:2]

    image = imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model=model)

    filtered_model = model.Filter(pyramid)

    detections = [d for i,d in itertools.izip(xrange(1), filtered_model.Parse(-1))]

    print(detections)

    detection_image = draw_detections (detections, image)
    imshow(detection_image)

def vis_test():
    model = LoadModel('tests/example.dpm')

    image = imread('tests/000034.jpg')
    pyramid = BuildPyramid(image, model=model)

    filtered_model = model.Filter(pyramid)

    detections_generator = filtered_model.Parse(-1)
    nms_generator = NMS(detections_generator, 0.3)
    detections = [d for i,d in itertools.izip(xrange(2), nms_generator)]

    print(detections)
    detection_image = draw_detections (detections, image)
    imshow(detection_image)

def hog_picture_test():
    model = LoadModel('tests/example.dpm')
    
    filter_w = model.start.rules[0].rhs[0].filter.blocklabel.w
    picture = hog_picture (filter_w, 20)
    imshow(picture)
