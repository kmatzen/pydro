from pydro._features import *

import scipy.misc
import numpy
import math
from collections import namedtuple

Level = namedtuple('Level', 'features,scale')

def PadLayer (layer, padx, pady):
    padded = numpy.pad(layer, ((pady+1,pady+1), (padx+1,padx+1), (0,0)), 'constant')
    padded[:pady+1,:,-1] = 1
    padded[-pady-1:,:,-1] = 1
    padded[:,:padx+1,-1] = 1
    padded[:,-padx-1:,-1] = 1
    return padded

def BuildPyramid (image, sbin, interval, extra_interval, padx, pady):
    if len(image.shape) == 2:
        image = numpy.dstack((image,image,image))

    sc = 2**(1.0/interval)
    max_scale = 1 + int(math.floor(math.log(min(image.shape[0:2])/(5.0*sbin))/math.log(sc)))

    def pyramid_generator():
        for i in xrange(interval):
            scale = 1/(sc**i)
            x = int(round(scale*image.shape[1]))
            y = int(round(scale*image.shape[0]))
            scaled = scipy.misc.imresize(image, (y,x))
            scaled_float = scaled.astype(numpy.float32)

            if extra_interval:
                yield Level(
                    features=PadLayer(ComputeFeatures(scaled_float, sbin/4), padx, pady), 
                    scale=4*scale,
                )

            yield Level(
                features=PadLayer(ComputeFeatures(scaled_float, sbin/2), padx, pady),
                scale=2*scale,
            )

            yield Level(
                features=PadLayer(ComputeFeatures(scaled_float, sbin), padx, pady),
                scale=scale,
            )

            for j in xrange(i+interval, max_scale, interval):
                scale /= 2
                x /= 2
                y /= 2
                scaled = scipy.misc.imresize(image, (y,x))
                scaled_float = scaled.astype(numpy.float32)
                
                yield Level(
                    features=PadLayer(ComputeFeatures(scaled_float, sbin), padx, pady),
                    scale=scale,
                )
     
    pyramid = list(pyramid_generator())
    pyramid.sort(key=lambda k: -k.scale)

    return pyramid
