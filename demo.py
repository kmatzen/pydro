import pydro.io
import pydro.features
import pydro.detection
import pydro.vis

import scipy.misc
import scipy.ndimage

import os
import urllib

model_filename = 'cars.dpm'
image_filename = 'demo.jpg'

if not os.path.exists(model_filename):
  print 'retrieving model'
  urllib.urlretrieve('http://foodnetwork.sndimg.com/content/dam/images/food/fullset/2013/5/2/2/FNM_060113-Guy-Cars-Spread_s4x3.jpg', image_filename)
if not os.path.exists(image_filename):
  print 'retrieving image'
  urllib.urlretrieve('https://s3.amazonaws.com/pydroapi/models/nyc3dcars/kitti_voc2007_nyc3dcars_wlssvm.dpm', 'cars.dpm')

print 'loading model'
pydro_model = pydro.io.LoadModel(model_filename)

print 'loading image'
image = scipy.misc.imread(image_filename)
print 'prefiltering image'
for i in xrange(3):
  image[:, :, i] = scipy.ndimage.filters.gaussian_filter(image[:, :, i], 3)
print 'resizing image'
image = scipy.misc.imresize(image, 0.2)

print 'building feature pyramid'
pyramid = pydro.features.BuildPyramid(image, model=pydro_model)

print 'filtering pyramid with model'
filtered_model = pydro_model.Filter(pyramid)

print 'generating parse trees'
# This produces a generator that yields all possible parse trees above some threshold.
threshold = -0.5
# You can also use the learned threshold in pydro_model.thresh
all_parse_trees = filtered_model.Parse(threshold)
# But you probably want to apply NMS to get a small set of non-overlapping detections.
nms_threshold = 0.3
nms_parse_trees = list(pydro.detection.NMS(all_parse_trees, nms_threshold))

for tree in nms_parse_trees:
  # The top node in each parse tree contains the bounding box and score.
  print 'x:{0}-{1}, y:{2}-{3}, score:{4}'.format(tree.x1, tree.x2, tree.y1, tree.y2, tree.s)
  # Metadata is stored in some nodes.  For example, this model also predicts orientation using
  # the mixture component from the DPM that had the highest response.
  print tree.child.rule.metadata

print 'visualize the detections (vis.png)'
# Object bounding boxes and part bounding boxes are drawn.
vis_image = pydro.vis.draw_detections(nms_parse_trees, image)
scipy.misc.imsave('vis.png', vis_image)
