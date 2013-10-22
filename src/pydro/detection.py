from pydro._detection import *
import multiprocessing
import itertools
from collections import namedtuple

__all__ = [
    'FilterPyramid',
    'FilterImage',
    'DeformationCost',
    'NMS',
    'Score',
]

Score = namedtuple('Score', 'score,scale')


def FilterPyramid(pyramid, filter, size):
    filtered = FilterImages(
        [level.features for level in pyramid.levels], filter, 0, size)

    for level in filtered:
        level.flags.writeable = False

    assert len(size) == len(filtered)
    assert len(pyramid.levels) == len(filtered)
    score = [
        Score(scale=level.scale, score=filtered)
        for level, filtered in itertools.izip(pyramid.levels, filtered)
    ]

    return score


def _intersection(detection1, detection2):
    return max(0, min(detection1.x2, detection2.x2) - max(detection1.x1, detection2.x1)) * \
        max(0, min(detection1.y2, detection2.y2)
            - max(detection1.y1, detection2.y1))


def _area(detection):
    return (detection.x2 - detection.x1) * (detection.y2 - detection.y1)


def _union(detection1, detection2):
    return _area(detection1) + _area(detection2) - _intersection(detection1, detection2)


def _overlap(detection1, detection2):
    return _intersection(detection1, detection2) / _union(detection1, detection2)


def NMS(detection_generator, threshold):
    accepted = []

    for detection in detection_generator:
        success = True
        for existing in accepted:
            if _overlap(detection, existing) > threshold:
                success = False
                break

        if success:
            accepted += [detection]
            yield detection
