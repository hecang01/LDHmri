import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 加载图形
image_dir = "D:\\DATA1\\MRIi"
all_images = os.listdir(image_dir)
X = []
y = []
for img in all_images:

    # 载入图形并重构
    full_path = os.path.join(image_dir, img)
    x = tf.io.read_file(full_path)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.resize(x, [128, 128])
    X.append(x)

    # 创建标签
    if img.endswith("1.png"):
        y.append(1)
    else:
        y.append(0)

# 列表转换为numpy
X = np.array(X)
y = np.array(y)

# 创建模型
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
# 丢弃率0.1
model.add(tf.keras.layers.Dropout(0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))

# 配置模型
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 训练模型
history = model.fit(X, y, batch_size=64, epochs=10)

# 保存模型
model.save("D:\\DATA1\\train\\MRI3.h5")
print("saved")

# 加载图形
image_dir1 = "D:\\DATA1\\test\\MRIi"
all_images = os.listdir(image_dir1)
X1 = []
y1 = []
for img in all_images:

    # 载入图形并重构
    full_path = os.path.join(image_dir1, img)
    x1 = tf.io.read_file(full_path)
    x1 = tf.image.decode_png(x1, channels=3)
    x1 = tf.image.resize(x1, [128, 128])
    X1.append(x1)

    # 创建标签
    if img.endswith("1.png"):
        y1.append(1)
    else:
        y1.append(0)

# 列表转换为numpy
X1 = np.array(X1)
y1 = np.array(y1)

# 测试
predictions = model.predict(X1)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.equal(predicted_classes, y1).mean()

# 输出准确率
print("Accuracy:", accuracy)

# 输出列表
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted classes:", predicted_classes)
