from pydro._detection import *

def FilterPyramid (pyramid, filter):
    for level in pyramid:
        yield Filter(level, filter)
