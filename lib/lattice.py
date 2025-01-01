import numpy as np

# import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from lib.common import rotate_right
from lib.detect_dot import detect_dot_at


# dots から格子の間隔を推定する
def infer_lattice_stride(dots):
    max_neightbor_distance = 64
    max_inclination = 1.2
    bucket_interval = 2

    # 点間のマンハッタン距離を計算
    buckets = defaultdict(list)
    # distance_vectors = []
    for i, dot1 in enumerate(dots):
        for j, dot2 in enumerate(dots):
            if i < j:  # 重複計算を避ける
                diff_x, diff_y = dot2[0] - dot1[0], dot2[1] - dot1[1]
                # X軸正方向が主軸となるように補正
                if np.abs(diff_x) < np.abs(diff_y):
                    diff_x, diff_y = diff_y, -diff_x
                if diff_x < 0:
                    diff_x, diff_y = -diff_x, -diff_y
                # マンハッタン距離と傾きが条件を満たす場合に記録
                dist = diff_x + np.abs(diff_y)
                if dist / diff_x < max_inclination and dist <= max_neightbor_distance:
                    bucket_index = (dist // bucket_interval) * bucket_interval
                    buckets[bucket_index].append(np.array([diff_x, diff_y]))

    if len(buckets) == 0:
        return (max_neightbor_distance * 0.8, 0)  # 適当なデフォルト値

    # 距離の頻度を解析し、最頻値を格子間隔とする
    max_bucket = max(buckets.items(), key=lambda x: len(x[1]))[1]
    return np.mean(np.array(max_bucket), axis=0)


# 始点から指定方向に向かって点を検出する
def detect_dots_toward(image, p0, direction, length=0, block_size=16):
    def in_bounds(p):
        n = block_size // 2
        return n <= p[0] < image.shape[1] - n and n <= p[1] < image.shape[0] - n

    detected_dots = []
    p1 = p0
    p2 = p0 + direction
    confident_dots = 0
    total_dots = 0
    max_iteration = 100
    while in_bounds(p2) and len(detected_dots) < max_iteration:
        detected_dot = detect_dot_at(image, p2, block_size=block_size)

        if detected_dot is not None:
            confident_dots += 1
        elif length > 0 and len(detected_dots) < length:
            detected_dot = p2  # use estimated point

        if detected_dot is not None:
            total_dots += 1
            detected_dots.append(detected_dot)
            p2 = detected_dot * 2 - p1
            p1 = detected_dot
        else:
            break

    confidence = confident_dots / total_dots if total_dots > 0 else 0
    return detected_dots, confidence


# 格子点をスキャンする
def scan_lattice_from(image, left_top_dot, lattice_stride):
    x_axis, _ = detect_dots_toward(image, left_top_dot, lattice_stride)
    y_axis, _ = detect_dots_toward(image, left_top_dot, rotate_right(lattice_stride))
    if len(x_axis) == 0 or len(y_axis) == 0:
        return None, None

    points = np.zeros((len(x_axis) + 1, len(y_axis) + 1, 2), dtype=np.float64)
    points[0, 0] = left_top_dot
    points[1:, 0] = np.resize(x_axis, points[1:, 0].shape)
    points[0, 1:] = np.resize(y_axis, points[0, 1:].shape)

    confidence = 1.0
    pending_confidence = 1.0
    garbages = 0
    for x in range(1, len(x_axis) + 1):
        y_scanned, local_confidence = detect_dots_toward(
            image,
            points[x, 0],
            points[x - 1, 1] - points[x - 1, 0],
            length=len(y_axis),
        )
        points[x, 1:] = np.resize(y_scanned, points[x, 1:].shape)
        if local_confidence < 0.2:
            garbages += 1
            pending_confidence = min(pending_confidence, local_confidence)
        else:
            confidence = min(min(local_confidence, confidence), pending_confidence)
            pending_confidence = 1.0
            garbages = 0

    if garbages > 0:
        points = points[:-garbages, :, :]

    return points, confidence


# 格子点をスキャンする
def scan_lattice(image, initial_dots, lattice_stride):
    sorted_dots = sorted(initial_dots, key=lambda x: np.linalg.norm(x))
    for left_top_dot in sorted_dots:
        lattice, confidence = scan_lattice_from(image, left_top_dot, lattice_stride)
        if lattice is not None:
            return lattice, confidence
    return None, None
