import Queue

import numpy
import scipy.misc

from pydro.core import *


def ShowDetections(trees, image):
    canvas = image.copy()

    colors = numpy.array([
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 0],
    ])

    for tree in trees:
        q = Queue.Queue()
        q.put((tree.child, 0))

        while not q.empty():
            node, depth = q.get()

            if isinstance(node, TreeNode):
                for child in node.children:
                    q.put((child, depth + 1))

            else:
                color = colors[depth, :]

                x1 = int(round(node.x1))
                y1 = int(round(node.y1))
                x2 = int(round(node.x2))
                y2 = int(round(node.y2))

                if x1 >= 0:
                    canvas[
                        max(0, y1):min(y2, canvas.shape[0] - 1) + 1, x1] = color
                if x2 < canvas.shape[1]:
                    canvas[
                        max(0, y1):min(y2, canvas.shape[0] - 1) + 1, x2] = color
                if y1 >= 0:
                    canvas[
                        y1, max(0, x1):min(x2, canvas.shape[1] - 1) + 1] = color
                if y2 < canvas.shape[0]:
                    canvas[
                        y2, max(0, x1):min(x2, canvas.shape[1] - 1) + 1] = color

    scipy.misc.imshow(canvas)
