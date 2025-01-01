import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

from common import script_based_path

model_file = script_based_path("digit_model.keras")
classes_per_digit = [1, 2, 4, 4, 4]
total_classes = sum(classes_per_digit)


# データを読み込む関数
def load_data(data_dir):
    images = []
    labels = []
    label = 0
    for index, classes in enumerate(classes_per_digit):
        folder_path = os.path.join(data_dir, str(index))
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # モノクロ画像を読み込む
            # img = cv2.resize(img, (28, 28))  # サイズを28x28に調整
            img = cv2.normalize(
                img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            for i in range(classes):
                images.append(img)
                labels.append(label + (i % classes))
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # hoge = False
        label += classes
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


# データをロード
data_dir = "data"  # データフォルダのパス
images, labels = load_data(data_dir)

# データを分割（80%訓練、20%テスト）
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
X_train = X_train.reshape(-1, 28, 28, 1)  # TensorFlow用の形状に変更
X_test = X_test.reshape(-1, 28, 28, 1)

# ラベルをOne-hotエンコード
y_train = tf.keras.utils.to_categorical(y_train, num_classes=total_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=total_classes)

# モデルを作成
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(
            total_classes, activation="softmax"
        ),  # クラス数に合わせる
    ]
)

# モデルをコンパイル
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# モデルを訓練
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# モデルを評価
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# モデルを保存
model.save(model_file)
