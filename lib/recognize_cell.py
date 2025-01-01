from collections import deque
import random
import cv2
import numpy as np

from lib.digit_model import (
    extract_digit_image,
    label_to_digit_and_rotation,
    batch_recognize_digits,
)


def draw_digit(dst_image, dst_points, digit, rotation):
    source_image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.putText(
        source_image, str(digit), (8, 48), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 2
    )

    for _ in range(rotation):
        source_image = cv2.rotate(source_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    src_points = np.array([[0, 0], [0, 64], [64, 64], [64, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    cv2.warpPerspective(
        source_image,
        matrix,
        (dst_image.shape[1], dst_image.shape[0]),
        dst=dst_image,
        borderMode=cv2.BORDER_TRANSPARENT,
    )


def recognize_cells(image, lattice, output_image):
    def cell_points(x, y):
        return np.array(
            [
                lattice[y, x],
                lattice[y, x + 1],
                lattice[y + 1, x + 1],
                lattice[y + 1, x],
            ],
            dtype=np.float32,
        )

    rows, cols = lattice.shape[:2]
    cell_images = []
    for y in range(rows - 1):
        for x in range(cols - 1):
            src_points = cell_points(x, y)
            cell_image = extract_digit_image(image, src_points)
            cell_images.append(cell_image)

    if len(cell_images) == 0:
        return None, 0

    predictions = batch_recognize_digits(cell_images)
    prediction_iter = iter(predictions)

    rotation_votes = np.array([0, 0, 0, 0])
    digits_image = np.zeros_like(output_image)
    result = np.zeros((rows - 1, cols - 1), dtype=np.int8)
    for y in range(rows - 1):
        for x in range(cols - 1):
            label, confidence = next(prediction_iter)
            digit, rotation_mask = label_to_digit_and_rotation(label)
            result[y, x] = digit

            if digit >= 0:
                # 回転角度の投票
                rotation = 0
                for r in range(4):
                    if rotation_mask & (1 << r):
                        rotation_votes[r] += 1
                        rotation = r
                        break

                # 数値をバッファに描画
                src_points = cell_points(x, y)
                draw_digit(digits_image, src_points, digit, rotation)

                # 訓練用にセル画像を保存
                if False:
                    for _ in range(rotation):
                        cell_image = cv2.rotate(cell_image, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(
                        f"tmp/{label}/{random.randint(100, 999)}.png", cell_image
                    )

    # 盤面の回転を補正
    field_rotation = max(range(4), key=lambda i: rotation_votes[i])
    result = np.transpose(result)
    result = np.rot90(result, -field_rotation)

    # 画面表示
    cv2.addWeighted(output_image, 1, digits_image, 0.4, 0, dst=output_image)
    cv2.imshow("Result", output_image)

    return result, field_rotation
