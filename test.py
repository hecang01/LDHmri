import os
import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model('D:\\DATA1\\train\\MRI\\test.h5')


# 加载数据集
def load_dataset(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        image = tf.io.read_file(filepath)
        image = tf.image.decode_png(image)
        image = tf.image.resize(image, (128, 128))
        images.append(image)
        filenames.append(filename)
    return images, filenames

# 正确率
correct_predictions = 0
total_predictions = 0

# 加载测试数据集
extra_images, extra_filenames = load_dataset('D:\\DATA1\\MRIi\\test')

# 将数据集转化为 TensorFlow Dataset 对象
extra_dataset = tf.data.Dataset.from_tensor_slices(extra_images)
extra_dataset = extra_dataset.batch(32)

# 预测
predictions = model.predict(extra_dataset)

# 输出预测结果
for prediction, filename in zip(predictions, extra_filenames):
    if prediction > 0.5:
        print(filename + " is positive")
        # 阳性正确
        if filename.endswith('1.png'):
            correct_predictions += 1
    else:
        print(filename + " is negative")
        # 阴性正确
        if filename.endswith('0.png'):
            correct_predictions += 1
    total_predictions += 1

# 计算准确率
accuracy = correct_predictions / total_predictions
print('accuracy: {:.2%}'.format(accuracy))
