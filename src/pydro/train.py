from pydro.core import TreeNode

import numpy

def BuildFeatureVector (model, detection):
    features = detection.child.symbol.GetFeatures (model, detection.child)
    return features

