import tensorflow as tf
import os

# 定义模型超参数
learning_rate = 0.001
num_epochs = 10
batch_size = 32


# 加载数据集
def load_dataset(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        # 解析文件名获取标签
        label = int(filename[-5])
        filepath = os.path.join(directory, filename)
        image = tf.io.read_file(filepath)
        image = tf.image.decode_png(image)
        image = tf.image.resize(image, (128, 128))
        images.append(image)
        labels.append(label)
    return images, labels


# 加载数据集
train_images, train_labels = load_dataset('D:\\DATA1\\MRIi')
val_images, val_labels = load_dataset('D:\\DATA1\\val\\MRIi')
test_images, test_labels = load_dataset('D:\\DATA1\\test\\MRIi')

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

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

# 评估模型
loss, acc = model.evaluate(test_dataset)
print("Loss: {}, Accuracy: {}".format(loss, acc))

# 保存模型
model.save('D:\\DATA1\\train\\MRI1.h5')
