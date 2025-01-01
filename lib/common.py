import os
import cv2
import numpy as np


def process_image_for_block(image, block_dim, f, offset=(0, 0), ubound=None):
    h, w = image.shape[:2]
    if ubound is not None:
        w, h = min(ubound[0], w), min(ubound[1], h)
    bw, bh = block_dim
    for y in range(offset[1], h, bh):
        for x in range(offset[0], w, bw):
            block = image[y : y + bh, x : x + bw]
            if block.shape[0] != bh or block.shape[1] != bw:
                continue
            result = f(block, x, y)
            if result == "stop":
                return


def sharpen_filter(image):
    # シャープ化カーネルを定義
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # シャープフィルタを適用
    return cv2.filter2D(image, -1, kernel)


def rotate_right(p):
    return np.array([-p[1], p[0]])


def is_full(c):
    return len(c) == c.maxlen


def script_based_path(file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, file)
