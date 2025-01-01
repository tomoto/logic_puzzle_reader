import cv2
import numpy as np
from scipy.spatial import distance
from lib.common import process_image_for_block


# ブロックがひとつの点のみを含むかどうか判定し、その座標を返す
def detect_dot_in_block(
    block, radius=5, focus_threshold=0.9, background_variance_threshold=10
):

    def find_centroid_and_focus(block):
        if block is None:
            return None

        min_val, max_val, _, _ = cv2.minMaxLoc(block)
        if max_val - min_val < background_variance_threshold:
            return None

        # ブロック全体の輝度を計算
        total_intensity = np.sum(block)
        # ブロック全体の輝度が 0 の場合は解なし
        if total_intensity == 0:
            return None

        rows, cols = block.shape

        # ブロックの重心の座標を計算
        centroid_x = np.sum(np.arange(cols) * np.sum(block, axis=0)) / total_intensity
        centroid_y = np.sum(np.arange(rows) * np.sum(block, axis=1)) / total_intensity

        # 重心を中心とした円形のマスクを作成
        mask = np.zeros_like(block, dtype=bool)
        for y in range(rows):
            for x in range(cols):
                if (x - centroid_x) ** 2 + (y - centroid_y) ** 2 < radius**2:
                    mask[y, x] = True

        # マスク内の輝度の合計を計算し、集中度(全体の輝度に対する割合)を計算
        focused_intensity = np.sum(block[mask])
        focused_ratio = focused_intensity / total_intensity

        # 計算結果を返す
        return (np.array([centroid_x, centroid_y]), focused_ratio)

    # ブロックの輝度を正規化し、二値化
    norm_block = cv2.normalize(
        block, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    _, binary_block = cv2.threshold(norm_block, 255 * 0.3, 255, cv2.THRESH_BINARY)
    # 二値化したブロックから重心と集中度を計算
    centroid_and_focus = find_centroid_and_focus(binary_block)
    # 集中度が閾値を越えていたら検出されたと見做す
    if centroid_and_focus:
        centroid, focus = centroid_and_focus
        if focus > focus_threshold:
            return centroid

    return None


def detect_dot_at(image, p, radius=5, block_size=16):
    # 点を整数値に変換し、そこを中心とするブロックの範囲を計算
    p = np.array(p, dtype=np.uint32)
    block_dim_half = np.array([block_size // 2, block_size // 2])
    p1, p2 = p - block_dim_half, p + block_dim_half

    # ブロックを切り出して検出処理を実行
    block = image[p1[1] : p2[1], p1[0] : p2[0]]

    detected_dot = detect_dot_in_block(block, radius)

    return p1 + detected_dot if detected_dot is not None else None


# 画像から点を検出する
def detect_dots(image, block_size=24, radius=5, max=50):
    detected_dots = []

    def process_block(block, x, y):
        detected_dot = detect_dot_in_block(block, radius)
        if detected_dot is not None:
            detected_dots.append(np.array([x, y]) + detected_dot)
        if len(detected_dots) >= max:
            return "stop"

    block_dim = np.array([block_size, block_size])
    half_block_dim = np.array([block_size // 2, block_size // 2])
    search_area = np.array([600, 400])
    process_image_for_block(image, block_dim, process_block, ubound=search_area)
    process_image_for_block(
        image, block_dim, process_block, ubound=search_area, offset=half_block_dim
    )

    # 近い点をマージ
    merge_distance = radius * 2
    tmp_dots = np.array(detected_dots)
    merged_dots = []
    while len(tmp_dots) > 0:
        dot = tmp_dots[0]
        distances = distance.cdist([dot], tmp_dots)
        close_dots = tmp_dots[distances.flatten() <= merge_distance]
        merged_dot = np.mean(close_dots, axis=0)
        merged_dots.append(merged_dot)
        tmp_dots = tmp_dots[distances.flatten() > merge_distance]

    return np.array(merged_dots)
