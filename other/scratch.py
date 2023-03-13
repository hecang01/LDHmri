import os
import xlwt

# 创建workbook和sheet对象
workbook = xlwt.Workbook()
sheet = workbook.add_sheet('Sheet1')

# 定义文件夹路径
folder_path = r"D:\\DATA1\\z"

# 获取该文件夹下的所有文件夹名称
folder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

# 将文件夹名称写入Excel表格
for i, folder_name in enumerate(folder_names):
    sheet.write(i, 0, folder_name)

# 保存Excel文件
workbook.save(r"D:\DATA1\z\folder_names.xls")
