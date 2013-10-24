"""Routines for training DPM."""

from pydro.core import Score
from pydro._train import compute_overlap

import numpy
import itertools
from collections import namedtuple
import scipy.misc
import scipy.optimize

__all__ = [
    'build_feature_vector',
    'get_positive_latent_features',
    'get_negative_latent_features',
    'compute_overlap',
    '__BBox__',
    'overlap_loss_adjustment',
    'optimize',
    'score_vector',
]

__BBox__ = namedtuple('__BBox__', 'x1,y1,x2,y2')
__TrainingExample__ = namedtuple(
    '__TrainingExample__', 'features,belief,loss,score')


def build_feature_vector(detection, belief, positive):
    """Takes a detection parse tree, extras relevant features,
       and constructs training example."""

    features = detection.child.symbol.GetFeatures(
        detection.model, detection.child)
    training_example = __TrainingExample__(
        features=features,
        belief=belief,
        loss=detection.loss if positive else 1.0,
        score=detection.s,
    )
    return training_example


def score_vector(entry):
    """Computes score for training example."""

    score = 0.0
    for block in entry.features:
        score += entry.features[block].flatten().T.dot(block.w.flatten())

    return score


def get_positive_latent_features(model, pyramid, belief_adjustment,
                                 loss_adjustment, num_loss):
    """Computes belief and num_loss loss adjusted detections for a
       positive training instance."""

    filtered_model_belief = model.Filter(
        pyramid, loss_adjustment=belief_adjustment)
    belief = [build_feature_vector(d, belief=True, positive=True)
              for d in itertools.islice(filtered_model_belief.Parse(-1), 1)
              ]

    positive_dummy = [__TrainingExample__(
        features={},
        belief=False,
        loss=1.0,
        score=0.0,
    )]

    filtered_model_loss = filtered_model_belief.Filter(
        loss_adjustment=loss_adjustment)
    loss = [build_feature_vector(d, belief=False, positive=True)
            for d in itertools.islice(filtered_model_loss.Parse(-1), num_loss)
            ]

    return belief + positive_dummy + loss


def get_negative_latent_features(model, pyramid, num_examples):
    """Mines image for num_examples false positive training examples."""

    filtered_model = model.Filter(pyramid)
    loss = [
        build_feature_vector(d, belief=False, positive=False)
        for d in itertools.islice(filtered_model.Parse(-1), num_examples)
    ]

    negative_dummy = [__TrainingExample__(
        features={},
        belief=True,
        loss=0.0,
        score=0.0,
    )]

    return loss + negative_dummy


def overlap_loss_adjustment(model, pyramid, threshold, value, rules, bbox):
    """Creates a loss adjustment function that considers bbox overlap."""

    def _overlap_loss_adjustment(rule, score):
        """A loss adjustment function that considers bbox overlap."""

        if rule not in rules:
            return score

        adjusted_score = []
        for level in score:
            scale = model.sbin / level.scale
            overlap = compute_overlap(
                bbox.x1, bbox.y1, bbox.x2, bbox.y2,
                rule.detwindow[0], rule.detwindow[1],
                level.score.shape[0], level.score.shape[1],
                scale,
                pyramid.pady + rule.shiftwindow[0],
                pyramid.padx + rule.shiftwindow[1],
                pyramid.image.shape[0], pyramid.image.shape[1]
            )

            loss = numpy.zeros(overlap.shape)
            loss[overlap < threshold] = value
            loss.flags.writeable = False
            adjusted_score += [
                Score(scale=level.scale, score=level.score + loss)]
            for level in adjusted_score:
                level.score.flags.writeable = False

        return adjusted_score

    return _overlap_loss_adjustment


def optimize(model, examples, svm_c):
    """Reoptimizes model given training examples."""

    blocks = model.GetBlocks()
    num_parms = 0
    block_sections = {}

    for block in blocks:
        block.w.flags.writeable = True
        end = num_parms + block.w.size
        block_sections[block] = (num_parms, end)
        num_parms = end

    initial_solution = numpy.zeros((num_parms,))
    for block in blocks:
        start, end = block_sections[block]
        initial_solution[start:end] = block.w.flatten()

    def _objective_function(solution):
        """Objective function for training."""

        gradient_packed = {block: numpy.zeros((block.w.size,))
                           for block in blocks}
        for block in blocks:
            start, end = block_sections[block]
            block.w[:] = solution[start:end].reshape(block.w.shape)

        objective_value = 0
        for example in examples:
            loss_adjusted_score = -numpy.inf
            max_nonbelief_score = -numpy.inf
            max_entry = None
            belief_entry = None

            for entry in example:
                score = score_vector(entry)

                loss_adjusted = score + entry.loss

                if entry.belief:
                    belief_score = score
                    belief_entry = entry
                elif loss_adjusted > max_nonbelief_score:
                    max_nonbelief_score = loss_adjusted

                if loss_adjusted > loss_adjusted_score:
                    max_entry = entry
                    loss_adjusted_score = loss_adjusted

            assert max_entry is not None
            assert belief_entry is not None

            objective_value += svm_c * (loss_adjusted_score - belief_score)

            if max_entry != belief_entry:
                for block in max_entry.features:
                    gradient_packed[block] += svm_c * \
                        max_entry.features[block].flatten()

                for block in belief_entry.features:
                    gradient_packed[block] -= svm_c * \
                        belief_entry.features[block].flatten()

        for block in blocks:
            objective_value += 0.5 * block.reg_mult * \
                block.w.flatten().T.dot(block.w.flatten())
            gradient_packed[block] += block.reg_mult * block.w.flatten()

        #f = ObjectiveFunction (examples)

        gradient = numpy.zeros(solution.shape)
        #Gradient (examples, gradient_packed)
        for block in blocks:
            start, end = block_sections[block]
            gradient[start:end] = gradient_packed[block]

        return objective_value, gradient

    solution, _, _ = scipy.optimize.fmin_l_bfgs_b(
        _objective_function, initial_solution)

    for block in blocks:
        start, end = block_sections[block]
        block.w[:] = solution[start:end].reshape(block.w.shape)
        block.w.flags.writeable = False
