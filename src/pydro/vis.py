import Queue

import numpy
from scipy.misc import imrotate

from pydro.core import TreeNode


def hog_picture(hog, resolution):
    glyph1 = numpy.zeros((resolution, resolution), dtype=numpy.uint8)
    glyph1[:, round(resolution / 2)-1:round(resolution / 2) + 1] = 255
    glyph = numpy.zeros((resolution, resolution, 9), dtype=numpy.uint8)
    glyph[:, :, 0] = glyph1
    for i in xrange(1, 9):
        glyph[:, :, i] = imrotate(glyph1, -i * 20)

    shape = hog.shape
    clamped_hog = hog.copy()
    clamped_hog[hog < 0] = 0
    image = numpy.zeros((resolution * shape[0], resolution * shape[1]), dtype=numpy.float32)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            for k in xrange(9):
                image[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution] = numpy.maximum(image[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution], clamped_hog[i, j, k] * glyph[:, :, k])

    return image


def draw_detections(trees, image):
    canvas = image.copy()

    colors = numpy.array([
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 0],
    ])
    colors.flags.writeable = False

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
    return canvas
