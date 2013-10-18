from pydro.core import TreeNode

import numpy

def BuildFeatureVector (model, detection, pyra):
    features = detection.child.symbol.GetFeatures (model, detection.child, pyra)
    return features

