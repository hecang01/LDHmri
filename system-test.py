import tensorflow as tf

# 检查tensorflow版本
print(tf.__version__)

# 查看当前配置
print(tf.config.list_physical_devices('GPU'))

# 运行一个简单的矩阵乘法，看看是否使用了GPU
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

print(c)
