from pydro.core import TreeNode
from pydro.features import BuildPyramid
from pydro._train import *

import numpy
import itertools

__all__ = [
    'BuildFeatureVector',
    'PositiveLatentFeatures',
    'NegativeLatentFeatures',
    'ComputeOverlap',
]

def BuildFeatureVector (detection):
    features = detection.child.symbol.GetFeatures (detection.model, detection.child)
    return features

def PositiveLatentFeatures (model, image, belief_adjustment, loss_adjustment, M, interval_fg):
    pyramid = BuildPyramid (image, model=model, interval=interval_fg)

    filtered_model_belief = model.Filter (pyramid, belief_adjustment)
    belief = BuildFeatureVector(filtered_model_belief.Parse(-1).next())
    
    filtered_model_loss = model.Filter (pyramid, loss_adjustment)
    loss = [BuildFeatureVector(d) for i,d in itertools.izip(xrange(M), filtered_model_loss.Parse(-1))]

    return belief, loss

def NegativeLatentFeatures (model, image, M, interval_bg):
    pyramid = BuildPyramid (image, model=model, interval=interval_bg)

    filtered_model = model.Filter(pyramid)
    return [BuildFeatureVector(d) for i,d in itertools.izip(xrange(M), filtered_model.Parse(-1))]

def OverlapLossAdjustment (model, threshold, value, symbols, bbox):
    def _overlap_loss_adjustment (symbol, score):
        if symbol not in symbols:
            return score

        adjusted_score = []
        for level in score:
            overlap = ComputeOverlap (bbox.x1, bbox.y1, bbox.x2, bbox.y2, 
                                      symbol.detwindow[0], symbol.detwindow[1], level.shape[0], level.shape[1],
                                      level.scale, model.pyramid.pady+symbol.shiftwindow[0], model.pyramid.padx+symbol.shiftwindow[1],
                                      pyramid.image.shape[0], pyramid.image.shape[1])
            loss = value * (overlap < threshold)
            adjusted_score += [Score(scale=level.scale, score=level.score+loss)]

        return adjusted_score

    return _overlap_loss_adjustment
