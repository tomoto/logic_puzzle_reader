import cv2
import tensorflow as tf
import numpy as np

from lib.common import script_based_path

# モデル(静的に保持される)
model_file = script_based_path("digit_model.keras")
digit_recognition_model = None


# 28x28の数字画像を image 上の src_point で囲まれた四角形から切り出す
def extract_digit_image(image, src_points):
    dst_points = np.array([[0, 0], [0, 28], [28, 28], [28, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (28, 28))


# 保存されたモデルを読み込む
def init_digit_recognition():
    global digit_recognition_model
    digit_recognition_model = tf.keras.models.load_model(model_file)


# 数字画像を認識する関数(単一; 遅いので使わない)
def recognize_digit(img):
    # サイズ調整と正規化
    if img.shape[0] != 28 or img.shape[1] != 28:
        img = cv2.resize(img, (28, 28))
    img = cv2.normalize(
        img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # モデルに入力できる形状に変更
    img = img.reshape(1, 28, 28, 1)  # (バッチサイズ, 高さ, 幅, チャンネル数)

    # 予測を実行
    predictions = digit_recognition_model.predict(img, verbose=0)

    # 最も確率が高いクラスを返す
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)

    return predicted_label, confidence


# 数字画像を認識する関数(バッチ処理で高速)
def batch_recognize_digits(images):
    preprocessed_images = []
    for img in images:
        if img.shape[0] != 28 or img.shape[1] != 28:
            img = cv2.resize(img, (28, 28))
        img = cv2.normalize(
            img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        # モデルに入力できる形状に変更
        img = img.reshape(28, 28, 1)
        preprocessed_images.append(img)

    batch = np.stack(preprocessed_images)

    # 予測を実行
    predictions = digit_recognition_model.predict(batch, verbose=0)

    return [(np.argmax(p), np.max(p)) for p in predictions]


# 認識後のラベルを数字と回転候補の組に変換
def label_to_digit_and_rotation(label):
    if label == 0:
        return (-1, 0x0F)
    elif label < 3:
        return (0, 0x05 << (label - 1))
    else:
        return ((label - 3) // 4 + 1, 1 << ((label - 3) % 4))
