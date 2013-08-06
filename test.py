#!/usr/bin/env python

import numpy
import pydro
import scipy.misc
import scipy.signal

image = scipy.misc.imread('/tmp/kmatzen/images/images/100243983_1c860ceb9a_o.jpg')
features = pydro.ComputeFeatures(image.astype(numpy.float32), 8)

patch = image[image.shape[0]/2-40:image.shape[0]/2+41,image.shape[1]/2-40:image.shape[1]/2+41,:]

filter = pydro.ComputeFeatures(patch.astype(numpy.float32), 8)

b = 1.2

filtered = pydro.Detect(features, filter, b)

print((filtered.min(), filtered.max()))

filtered_correct = numpy.sum(numpy.dstack(scipy.signal.correlate2d(features[:,:,i], filter[:,:,i], mode='same') for i in xrange(filter.shape[2])), axis=2) - b
filtered_correct[:3,:] = -2
filtered_correct[-4:,:] = -2
filtered_correct[:,:3] = -2
filtered_correct[:,-4:] = -2

scipy.misc.imshow(numpy.abs(filtered - filtered_correct) > 1e-3)
