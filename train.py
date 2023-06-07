import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义模型超参数
# 学习率
learning_rate = 0.001
# 迭代次数
num_epochs = 30
# 批大小
batch_size = 64


# 加载数据集
def load_dataset(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        # 解析文件名获取标签
        # 除了扩展名最后一位数字
        label = int(filename[-5])
        filepath = os.path.join(directory, filename)
        image = tf.io.read_file(filepath)
        # 图片解码
        image = tf.image.decode_png(image)
        image = tf.image.resize(image, (128, 128))
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)


# 加载数据集
# 训练集
train_images, train_labels = load_dataset('D:\\DATA1\\MRIi\\train')
# 验证集
val_images, val_labels = load_dataset('D:\\DATA1\\MRIi\\val')
# 测试集
test_images, test_labels = load_dataset('D:\\DATA1\\MRIi\\test')

# 将数据集转化为 TensorFlow Dataset 对象
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# 打乱并分批次
train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 创建数据增强器
datagen = ImageDataGenerator(
    rotation_range=20,  # 随机旋转角度范围20°
    width_shift_range=0.1,  # 随机水平平移范围10%
    height_shift_range=(0, 0.1),  # 随机垂直平移范围向上10%
    shear_range=0.2,  # 随机剪切强度
    zoom_range=(0.8, 1.05),  # 随机缩放范围80-105%
    horizontal_flip=True,  # 水平翻转
    vertical_flip=False  # 垂直翻转
)

# 创建用于增强阳性样本的数据生成器
positive_indices = np.where(train_labels == 1)[0]
positive_images = train_images[positive_indices]

desired_samples = 1000
augmented_positive_images = []
iterations = int(np.ceil(desired_samples / batch_size))

for _ in range(iterations):
    augmented_images = datagen.flow(positive_images, batch_size=batch_size)
    augmented_positive_images.extend(augmented_images[0])

augmented_positive_images = np.array(augmented_positive_images[:desired_samples])

positive_generator = datagen.flow(
    positive_images, train_labels[positive_indices], batch_size=batch_size, shuffle=True
)

# 设置阴性样本和阳性样本的权重
weight_neg = 1.0  # 阴性样本的权重
weight_pos = 3.0  # 阳性样本的权重
class_weights = {0: weight_neg, 1: weight_pos}

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 训练模型，包括阳性样本的数据增强
model.fit(
    datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=num_epochs,
    steps_per_epoch=len(train_images) // batch_size,
    validation_data=(val_images, val_labels)
)

# 评估模型
loss, acc = model.evaluate(test_dataset)
print("Loss: {}, Accuracy: {}".format(loss, acc))

# 保存模型
model.save('D:\\DATA1\\train\\MRI\\test.h5')
