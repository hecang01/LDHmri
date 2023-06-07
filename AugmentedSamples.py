import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# 定义模型超参数
# 学习率
learning_rate = 0.001
# 迭代次数
num_epochs = 120
# 批大小
batch_size = 128


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

# 创建 ImageDataGenerator 对象并设置增强参数
datagen = ImageDataGenerator(
    rotation_range=20,  # 随机旋转角度范围20°
    width_shift_range=0.1,  # 随机水平平移范围10%
    height_shift_range=(0, 0.1),  # 随机垂直平移范围向上10%
    shear_range=0.2,  # 随机剪切强度
    zoom_range=(0.8, 1.05),  # 随机缩放范围10%
    horizontal_flip=True,  # 水平翻转
    vertical_flip=False  # 垂直翻转
)

train_images_positive = train_images[train_labels == 1]
train_images_positive = train_images_positive.reshape((-1, 128, 128, 3))

# 设置生成的批次大小和生成样本的数量
desired_samples = 1000  # 生成的增强样本数量

# 创建目录用于保存增强样本
save_dir = 'D:/DATA1/AugmentedSamples'
os.makedirs(save_dir, exist_ok=True)

# 生成增强样本并保存
counter = 0

for batch_images in datagen.flow(train_images_positive, batch_size=32):
    for image in batch_images:
        filename = f'aug_{counter}.png'
        save_path = os.path.join(save_dir, filename)
        tf.keras.preprocessing.image.save_img(save_path, image)

        counter += 1
        if counter >= desired_samples:
            break
    if counter >= desired_samples:
        break
