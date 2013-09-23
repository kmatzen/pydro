from pydro._features import *

import numpy
import math
from collections import namedtuple
import sys

Level = namedtuple('Level', 'features,scale')

def BuildPyramid (image, sbin, interval, extra_interval, padx, pady):
    if len(image.shape) == 2:
        image = numpy.dstack((image,image,image))
    image = image.astype(numpy.float32)

    sc = 2**(1.0/interval)
    max_scale = 1 + int(math.floor(math.log(min(image.shape[0:2])/(5.0*sbin))/math.log(sc)))

    def pyramid_generator():
        for i in xrange(interval):
            scale = 1/(sc**i)
            x = int(round(image.shape[1]*scale))
            y = int(round(image.shape[0]*scale))
            sys.stdout.flush()
            scaled = ResizeImage(image, y,x)

            if extra_interval:
                yield Level(
                    features=ComputeFeatures(scaled, sbin/4, padx, pady), 
                    scale=4*scale,
                )

            yield Level(
                features=ComputeFeatures(scaled, sbin/2, padx, pady),
                scale=2*scale,
            )

            yield Level(
                features=ComputeFeatures(scaled, sbin, padx, pady),
                scale=scale,
            )

            for j in xrange(i+interval, max_scale, interval):
                scale *= 0.5
                x = int(round(x*0.5))
                y = int(round(y*0.5))
                scaled = ResizeImage(image, y,x)
                
                yield Level(
                    features=ComputeFeatures(scaled, sbin, padx, pady),
                    scale=scale,
                )
     
    pyramid = list(pyramid_generator())
    pyramid.sort(key=lambda k: -k.scale)

    return pyramid
