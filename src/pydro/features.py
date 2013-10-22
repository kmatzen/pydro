from pydro._features import *

import numpy
import math
from collections import namedtuple
import sys

Level = namedtuple('Level', 'features,scale')
Pyramid = namedtuple('Pyramid', 'levels,image,pady,padx,sbin,interval')


def BuildPyramid(image, model=None, sbin=None, interval=None, extra_octave=None, padx=None, pady=None):
    if sbin is None:
        sbin = model.sbin
    if interval is None:
        interval = model.interval
    if extra_octave is None:
        extra_octave = model.features.extra_octave
    if padx is None:
        padx = model.maxsize[1]
    if pady is None:
        pady = model.maxsize[0]

    if len(image.shape) == 2:
        image = numpy.dstack((image, image, image))
    image = image.astype(numpy.float32)
    image.flags.writeable = False

    sc = 2 ** (1.0 / interval)
    max_scale = 1 + \
        int(math.floor(
            math.log(min(image.shape[0:2]) / (5.0 * sbin)) / math.log(sc)))

    def level_generator():
        for i in xrange(interval):
            scale = 1 / (sc ** i)
            x = int(round(image.shape[1] * scale))
            y = int(round(image.shape[0] * scale))
            sys.stdout.flush()
            scaled = ResizeImage(image, y, x)

            if extra_octave:
                yield Level(
                    features=ComputeFeatures(
                        scaled, sbin / 4, padx + 1, pady + 1),
                    scale=4 * scale,
                )

            yield Level(
                features=ComputeFeatures(scaled, sbin / 2, padx + 1, pady + 1),
                scale=2 * scale,
            )

            yield Level(
                features=ComputeFeatures(scaled, sbin, padx + 1, pady + 1),
                scale=scale,
            )

            for j in xrange(i + interval, max_scale, interval):
                scale *= 0.5
                x = int(round(x * 0.5))
                y = int(round(y * 0.5))
                scaled = ResizeImage(image, y, x)

                yield Level(
                    features=ComputeFeatures(scaled, sbin, padx + 1, pady + 1),
                    scale=scale,
                )

    levels = list(level_generator())
    levels.sort(key=lambda k: -k.scale)

    pyramid = Pyramid(
        levels=levels,
        pady=pady,
        padx=padx,
        sbin=sbin,
        interval=interval,
        image=image,
    )

    return pyramid
