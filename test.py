import os
import tensorflow as tf
from sklearn import metrics

# 加载模型
model = tf.keras.models.load_model('D:\\DATA1\\train\\MRI\\test.h5')


# 加载数据集
def load_dataset(directory):
    images = []
    filenames = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        image = tf.io.read_file(filepath)
        image = tf.image.decode_png(image)
        image = tf.image.resize(image, (128, 128))
        images.append(image)
        filenames.append(filename)
        labels.append(int(filename[-5]))
    return images, filenames, labels


# 正确率
correct_predictions = 0
total_predictions = 0

# 加载测试数据集
test_images, test_filenames, test_labels = load_dataset('D:\\DATA1\\MRIi\\test')

# 将数据集转化为 TensorFlow Dataset 对象
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(32)

# 预测
predictions = model.predict(test_dataset)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]

# 输出预测结果
for prediction, filename in zip(predictions, test_filenames):
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

# 计算敏感性（召回率）
sensitivity = metrics.recall_score(test_labels, binary_predictions)
print('Sensitivity: {:.2%}'.format(sensitivity))

# 计算特异性
specificity = metrics.recall_score(test_labels, binary_predictions, pos_label=0)
print('Specificity: {:.2%}'.format(specificity))

# 计算阳性预测值（精确率）
ppv = metrics.precision_score(test_labels, binary_predictions)
print('PPV: {:.2%}'.format(ppv))

# 计算阴性预测值
npv = metrics.precision_score(test_labels, binary_predictions, pos_label=0)
print('NPV: {:.2%}'.format(npv))

# 计算AUC
auc = metrics.roc_auc_score(test_labels, predictions)
print('AUC: {:.2%}'.format(auc))