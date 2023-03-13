import os
import tensorflow as tf

# 定义模型超参数
# 学习率
learning_rate = 0.001
# 迭代次数
num_epochs = 20
# 批大小
batch_size = 320
# 处理次数
turn = 10


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
    return images, labels


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

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# 训练次数
count = 0
# 正确率
acc_all = 0

# 循环训练
for i in range(turn):
    # 训练模型
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

    # 评估模型
    loss, acc = model.evaluate(test_dataset)
    print("Loss: {}, Accuracy: {}".format(loss, acc))

    # 保存模型
    count += 1
    model_filename = 'model_{0}_{1}_{2}_{3}.h5'.format(str(num_epochs), str(batch_size), str(count),
                                                       str(round(acc, 4)))
    model_path = os.path.join('D:\\DATA1\\train\\MRI', model_filename)
    model.save(model_path)

    acc_all += acc

    print(count)

# 输出准确率
print(acc_all / turn)
