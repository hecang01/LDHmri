import os
import shutil
import random

# 定义主目录和目标文件夹
MRI_DIR = "D:\\DATA1\\MRIi\\mri"
TRAIN_DIR = "D:\\DATA1\\MRIi\\train"
TEST_DIR = "D:\\DATA1\\MRIi\\test"
VAL_DIR = "D:\\DATA1\\MRIi\\val"

# 确保目标文件夹存在
for folder in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 获取mri文件夹内的所有文件名
files = os.listdir(MRI_DIR)

# 按照文件名连续的12个文件为一组进行分组
groups = []
for i in range(0, len(files), 12):
    group = files[i:i + 12]
    groups.append(group)

# 遍历每个组，按照6:2:2的比例随机复制到train、test、val文件夹
for group in groups:
    # 随机选择6个文件复制到train文件夹
    train_files = random.sample(group, 6)
    for train_file in train_files:
        src = os.path.join(MRI_DIR, train_file)
        dst = os.path.join(TRAIN_DIR, train_file)
        shutil.copy(src, dst)
        # 随机选择2个文件复制到test文件夹
    test_files = random.sample(group, 2)
    for test_file in test_files:
        src = os.path.join(MRI_DIR, test_file)
        dst = os.path.join(TEST_DIR, test_file)
        shutil.copy(src, dst)
        # 随机选择2个文件复制到val文件夹
    val_files = random.sample(group, 2)
    for val_file in val_files:
        src = os.path.join(MRI_DIR, val_file)
        dst = os.path.join(VAL_DIR, val_file)
        shutil.copy(src, dst)