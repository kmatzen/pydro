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
    'ScoreVector',
]

BBox = namedtuple('BBox', 'x1,y1,x2,y2')
TrainingExample = namedtuple('TrainingExample', 'features,belief,loss,detection')

def BuildFeatureVector (detection, belief, positive):
    features = detection.child.symbol.GetFeatures (detection.model, detection.child)
    training_example = TrainingExample (
        features=features,
        belief=belief,
        loss=detection.loss if positive else 1.0,
        detection=detection,
    )
    return training_example

def ScoreVector (entry):
    score = 0.0
    for block in entry.features:
        score += entry.features[block].flatten().T.dot(block.w.flatten())

    return score

def PositiveLatentFeatures (model, pyramid, belief_adjustment, loss_adjustment, M):
    filtered_model_belief = model.Filter (pyramid, belief_adjustment)
    belief = [BuildFeatureVector(d, belief=True, positive=True) for i,d in itertools.izip(xrange(1), filtered_model_belief.Parse(-numpy.inf))]

    positive_dummy = [TrainingExample (
        features={},
        belief=False,
        loss=1.0,
        detection=None,
    )]
    
    filtered_model_loss = model.Filter (pyramid, loss_adjustment)
    loss = [BuildFeatureVector(d, belief=False, positive=True) for i,d in itertools.izip(xrange(M), filtered_model_loss.Parse(-1))]

    return belief + positive_dummy + loss

def NegativeLatentFeatures (model, pyramid, M):
    filtered_model = model.Filter(pyramid)
    loss = [BuildFeatureVector(d, belief=False, positive=False) for i,d in itertools.izip(xrange(M), filtered_model.Parse(-numpy.inf))]

    negative_dummy = [TrainingExample (
        features={},
        belief=True,
        loss=0.0,
        detection=None,
    )]

    return loss + negative_dummy

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

        f = 0
        for example in examples:
            V = -numpy.inf
            max_nonbelief_score = -numpy.inf
            I = None
            belief_I = None

            for entry in example:
                score = ScoreVector (entry)

                loss_adjusted = score + entry.loss

                if entry.belief:
                    belief_score = score
                    belief_I = entry
                elif loss_adjusted > max_nonbelief_score:
                    max_nonbelief_score = loss_adjusted

                if loss_adjusted > V:
                    I = entry 
                    V = loss_adjusted

            assert I is not None
            assert belief_I is not None

            C = 0.001
            f += C * (V - belief_score)

            if I != belief_I:
                for block in I.features:
                    g_packed[block] += C*I.features[block].flatten()

                for block in belief_I.features:
                    g_packed[block] -= C*belief_I.features[block].flatten()

        for block in blocks:
            f += 0.5 * block.reg_mult * block.w.flatten().T.dot(block.w.flatten())
            g_packed[block] += block.reg_mult * block.w.flatten()

        #f = ObjectiveFunction (examples)

        g = numpy.zeros(x.shape)
        #Gradient (examples, g_packed)
        for block in blocks:
            start, end = block_sections[block]
            g[start:end] = g_packed[block]

        print(f)

        return f, g

    x, f, d = scipy.optimize.fmin_l_bfgs_b (_objective_function, x0)

    for block in blocks:
        start, end = block_sections[block]
        block.w[:] = x[start:end].reshape(block.w.shape)

    print(d)
