from pydro._features import *

import scipy.misc
import numpy
import math
from collections import namedtuple

Level = namedtuple('Level', 'features,scale')

def BuildPyramid (image, sbin, interval, extra_interval):
    if len(image.shape) == 2:
        image = numpy.dstack((image,image,image))

    sc = 2**(1.0/interval)
    max_scale = 1 + int(math.floor(math.log(min(image.shape[0:2])/(5*sbin))/math.log(sc)))

    def pyramid_generator():
        for i in xrange(interval):
            scaled = scipy.misc.imresize(image, 1/sc**i)
            scaled_float = scaled.astype(numpy.float32)

            if extra_interval:
                yield Level(
                    features=ComputeFeatures(scaled_float, sbin/4), 
                    scale=4/sc**i,
                )

            yield Level(
                features=ComputeFeatures(scaled_float, sbin/2),
                scale=2/sc**i,
            )

            yield Level(
                features=ComputeFeatures(scaled_float, sbin),
                scale=1/sc**i,
            )

            scale = 1/sc**i
            for j in xrange(i+interval, max_scale, interval):
                scaled = scipy.misc.imresize(scaled, 0.5)
                scale *= 0.5
                scaled_float = scaled.astype(numpy.float32)
                
                yield Level(
                    features=ComputeFeatures(scaled_float, sbin),
                    scale=scale,
                )
     
    pyramid = list(pyramid_generator())
    pyramid.sort(key=lambda k: -k.scale)

    return pyramid
