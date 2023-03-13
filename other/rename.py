import os

# 设定文件夹
main_dir = 'D:\\DATA1\\MRIi'

# 下级目录
for dirname in os.listdir(main_dir):
    subdir = os.path.join(main_dir, dirname)

    # 下级目录文件夹
    for filename in os.listdir(subdir):
        # 文件路径
        filepath = os.path.join(subdir, filename)

        # 重命名加上文件夹名
        new_filename = dirname + '_' + filename
        new_filepath = os.path.join(subdir, new_filename)
        os.rename(filepath, new_filepath)

        # 移动文件
        final_filepath = os.path.join(main_dir, new_filename)
        os.rename(new_filepath, final_filepath)
