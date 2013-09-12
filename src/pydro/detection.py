from pydro._detection import *

def DetectPyramid (pyramid, filter):
    for level in pyramid:
        yield Detect(level, filter)
