from pydro._detection import *

__all__ = ['FilterPyramid', 'FilterImage', 'DeformationCost']

def FilterPyramid (pyramid, filter):
    return [FilterImage(level.features, filter) for level in pyramid]


