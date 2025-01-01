from collections import deque
import sys
import cv2
import numpy as np

from lib import output
from lib.common import is_full
from lib.detect_dot import detect_dots
from lib.lattice import infer_lattice_stride, scan_lattice
from lib.digit_model import init_digit_recognition
from lib.recognize_cell import recognize_cells


# fancy settings
input = "camera"  # カメラから入力
output_mode = "solver_to_image"  # 読み取った盤面を解いて画面に重ね書きし、さらに盤面と画像を out ディレクトリ下に保存
# output_mode = "solver_to_console"  # 読み取った盤面を解いてコンソールに出力
# output_mode = "file"  # 読み取った盤面を out ディレクトリ下にファイル出力

# safe settings
# input = "test_input.jpg"  # 画像ファイルから入力
# output_mode = "stdout"  # 読み取った盤面を標準出力に出力


def send_result(result, image, lattice, rotation):
    if output_mode == "solver_to_console":
        output.solver_to_console(result)
    elif output_mode == "solver_to_image":
        output.solver_to_image(result, image, lattice, rotation)
    elif output_mode == "file":
        output.to_file(result)
    else:  # stdout
        output.print_result(sys.stdout, result)


# 処理本体
def process(original_image, queue):
    # グレースケールに変換し、白黒反転
    grayed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayed_image = cv2.bitwise_not(grayed_image)

    # 初期点を検出
    initial_dots = detect_dots(grayed_image)

    # 格子の間隔を推定
    lattice_stride = infer_lattice_stride(initial_dots)

    # 格子点をスキャン
    lattice, confidence = scan_lattice(grayed_image, initial_dots, lattice_stride)

    # 表示画面を作成
    output_image = original_image.copy()

    # 初期点を赤で描画
    for d in np.array(initial_dots, dtype=np.uint32):
        cv2.circle(output_image, d, 4, (0, 0, 255), 1)

    # 格子が検出されなかったらそのまま表示して戻る
    if lattice is None or lattice.shape[0] < 5 or lattice.shape[1] < 5:
        cv2.imshow("Result", output_image)
        return

    confident_enough = confidence > 0.6

    # 格子を描画(確信度に応じて色を変える)
    confidence_color = (0, 255, 0) if confident_enough else (0, 255, 255)
    for d in np.array(lattice.reshape(-1, 2), dtype=np.uint32):
        cv2.circle(output_image, d, 4, confidence_color, 1)

    # この時点で表示
    cv2.imshow("Result", output_image)

    # 確信度が高ければキューに追加し、安定した結果が得られたら解析に移る
    if confident_enough:
        queue.append((lattice, confidence, grayed_image, output_image))
        if is_full(queue) and all(item[0].shape == queue[0][0].shape for item in queue):
            best = max(queue, key=lambda item: item[1])
            best_lattice, _, best_image, best_output_image = best
            result, rotation = recognize_cells(
                best_image, best_lattice, best_output_image
            )
            if result is not None:
                send_result(result, best_output_image, best_lattice, rotation)
                queue.clear()


init_digit_recognition()

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

if input == "camera":
    queue = deque(maxlen=8)
    cap = cv2.VideoCapture(0)
    # resolution = (1280, 720)
    resolution = (1920, 1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process(frame, queue)
        if cv2.waitKey(1) >= 0:
            break

    cap.release()
else:
    queue = deque(maxlen=1)
    original_image = cv2.imread(input)
    cv2.imshow("Result", original_image)
    process(original_image, queue)
    while cv2.waitKey(1000) < 0:
        pass

cv2.destroyAllWindows()
