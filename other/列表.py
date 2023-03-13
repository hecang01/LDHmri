import os
import pandas as pd

# 获取文件夹中的所有文件名
data_path = 'D:\\DATA1\\train\\MRI'
file_names = os.listdir(data_path)

# 创建一个DataFrame对象，并将文件名存入其中
df = pd.DataFrame({'File Name': file_names})

# 删除DataFrame对象中的扩展名
# df['File Name'] = df['File Name'].str[:-8]

# 删除重复项
# df.drop_duplicates(inplace=True)

# 将最终处理好的数据输出为excel文件
output_path = 'D:\\DATA1\\train\\MRItrain.xlsx'
df.to_excel(output_path, index=False)
