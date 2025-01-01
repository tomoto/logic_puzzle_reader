import platform
import subprocess
from datetime import datetime

import cv2
import numpy as np


solver_exe = "./bin/slilin" + (".exe" if platform.system() == "Windows" else "")
output_dir = "./out"


def generate_out_file_name(extension):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{output_dir}/Result-{timestamp}.{extension}"


def print_result(f, result):
    f.write(f"# Captured at {datetime.now()}\n")
    symbols = "-0123"
    for row in result:
        f.write("".join(symbols[x + 1] for x in row) + "\n")


def to_file(result):
    file = generate_out_file_name("txt")
    with open(file, "w") as f:
        print_result(f, result)


def solver_to_console(result):
    # ソルバーに問題を一方的に渡す; 出力はソルバーそのまま
    solver = [solver_exe]
    child = subprocess.Popen(solver, stdin=subprocess.PIPE, text=True)
    print_result(child.stdin, result)
    child.stdin.close()
    child.wait()


def solver_to_image(result, image, lattice, rotation):
    def is_empty_cell(c):
        return c in (" ", "x")

    # 問題をファイルに保存
    to_file(result)

    # ソルバーに問題を渡し、出力を受け取る
    solver = [solver_exe, "-f", "plain", "-r"]
    child = subprocess.Popen(
        solver,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    print_result(child.stdin, result)
    child.stdin.close()
    solution = child.stdout.readlines()

    # 出力画像に線をオーバーレイ
    lattice = np.array(np.rot90(lattice, rotation), dtype=np.int32)
    cols, rows = lattice.shape[:2]
    line_color = (20, 20, 20)
    for y in range(rows):
        for x in range(cols - 1):
            if not is_empty_cell(solution[y * 2][x * 2 + 1]):
                cv2.line(
                    image,
                    lattice[x, y],
                    lattice[x + 1, y],
                    line_color,
                    2,
                )
    for x in range(cols):
        for y in range(rows - 1):
            if not is_empty_cell(solution[y * 2 + 1][x * 2]):
                cv2.line(
                    image,
                    lattice[x, y],
                    lattice[x, y + 1],
                    line_color,
                    2,
                )

    # 正立像を保存
    saved_image = image
    for _ in range(rotation):
        saved_image = cv2.rotate(saved_image, cv2.ROTATE_90_CLOCKWISE)
    saved_image_file = generate_out_file_name("jpg")
    cv2.imwrite(saved_image_file, saved_image)

    # 2秒待って次に進む
    cv2.imshow("Result", image)
    cv2.waitKey(2000)
