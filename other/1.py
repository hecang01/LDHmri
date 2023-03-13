import os

# 遍历当前目录下的所有文件
image_dir = "D:\\DATA1\\MRIi"
for filename in os.listdir(image_dir):
    # 如果文件名既不是 1.png 结尾也不是 0.png 结尾，输出文件名
    if not filename.endswith('1.png') and not filename.endswith('0.png'):
        print(filename)

