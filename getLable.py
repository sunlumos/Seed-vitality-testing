import pandas as pd

# 读取Excel文件
data = pd.read_excel('D:\S\start\code\seeds\\test.xlsx', header=None)

# !二分类
# # 按行处理数据
# output = []
# for index, row in data.iterrows():
#     row_data = [1 if value != 0 else 0 for value in row]
#     output.extend(row_data)

# # 将结果输出到新的Excel文件
# output_data = pd.DataFrame({'Result': output})
# output_data.to_excel('D:\S\start\code\seeds\output_file.xlsx', index=False)

# !三分类
# 0：未发芽，1：发芽天数>5天，2：发芽天数<=5天
# 处理数据
processed_data = []
for index, row in data.iterrows():
    for value in row:
        if value > 5:
            processed_data.append(1)
        elif 0 < value <= 5:
            processed_data.append(2)
        else:
            processed_data.append(0)

# 创建DataFrame并保存为Excel文件
processed_df = pd.DataFrame(processed_data, columns=['Processed'])
processed_df.to_excel('output.xlsx', index=False)  # 替换 'output.xlsx' 为输出的Excel文件路径