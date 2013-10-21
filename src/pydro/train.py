from pydro.core import TreeNode, Score
from pydro.features import BuildPyramid
from pydro._train import *

import numpy
import itertools
from collections import namedtuple
import scipy.misc
import scipy.optimize

__all__ = [
    'BuildFeatureVector',
    'PositiveLatentFeatures',
    'NegativeLatentFeatures',
    'ComputeOverlap',
    'BBox',
    'OverlapLossAdjustment',
    'Optimize',
]

BBox = namedtuple('BBox', 'x1,y1,x2,y2')

def BuildFeatureVector (detection):
    features = detection.child.symbol.GetFeatures (detection.model, detection.child)
    return features, detection

def PositiveLatentFeatures (model, pyramid, belief_adjustment, loss_adjustment, M):
    filtered_model_belief = model.Filter (pyramid, belief_adjustment)
    belief = [BuildFeatureVector(d) for i,d in itertools.izip(xrange(1), filtered_model_belief.Parse(-1))]
    
    filtered_model_loss = model.Filter (pyramid, loss_adjustment)
    loss = [BuildFeatureVector(d) for i,d in itertools.izip(xrange(M), filtered_model_loss.Parse(-1))]

    return belief, loss

def NegativeLatentFeatures (model, pyramid, M):
    filtered_model = model.Filter(pyramid)
    return [BuildFeatureVector(d) for i,d in itertools.izip(xrange(M), filtered_model.Parse(-1))]

def OverlapLossAdjustment (model, pyramid, threshold, value, rules, bbox):
    def _overlap_loss_adjustment (rule, score):
        if rule not in rules:
            return score

        adjusted_score = []
        for level in score:
            scale = model.sbin / level.scale
            overlap = ComputeOverlap (bbox.x1, bbox.y1, bbox.x2, bbox.y2, 
                                      rule.detwindow[0], rule.detwindow[1], level.score.shape[0], level.score.shape[1],
                                      scale, pyramid.pady+rule.shiftwindow[0], pyramid.padx+rule.shiftwindow[1],
                                      pyramid.image.shape[0], pyramid.image.shape[1])
            loss = numpy.zeros(overlap.shape)
            loss[overlap < threshold] = value
            adjusted_score += [Score(scale=level.scale, score=level.score+loss)]

        return adjusted_score

    return _overlap_loss_adjustment

def Optimize (model, examples):
    blocks = model.GetBlocks()
    nParams = 0
    block_sections = {}

    for block in blocks:
        end = nParams + block.w.size
        block_sections[block] = (nParams,end)
        nParams = end

    x0 = numpy.zeros((nParams,))
    for block in blocks:
        start, end = block_sections[block]
        x0[start:end] = block.w.flatten()

    def _objective_function (x, *args):
        g_packed = { block : numpy.zeros((block.w.size,)) for block in blocks }
        for block in blocks:
            start, end = block_sections[block]
            block.w[:] = x[start:end].reshape(block.w.shape)

        f = ObjectiveFunction (examples)

        g = numpy.zeros(x.shape)
        Gradient (examples, g_packed)
        for block in blocks:
            start, end = block_sections[block]
            g[start:end] = g_packed[block]

        return f, g

    x, f, d = scipy.optimize.fmin_l_bfgs_b (_objective_function, x0)

    for block in blocks:
        start, end = block_sections[block]
        block.w[:] = x[start:end].reshape(block.w.shape)

    print(d)
