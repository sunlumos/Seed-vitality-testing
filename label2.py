import math
import pandas as pd
# !用于表格划分数据集

filename = 'D:\S\start\code\seeds\\firstdata\yongyou9.csv'
df = pd.read_csv(filename, header=None)

labels = df.iloc[:, -1]
grouped = df.groupby(labels)

datasets = []
for _, group in grouped:
    datasets.append(group)

train_ratio, val_ratio, test_ratio = (3/4,1/8,1/8)
train_data = pd.DataFrame()
val_data = pd.DataFrame()
test_data = pd.DataFrame()

for dataset in datasets:
    n = len(dataset)
    train_size = round(n * train_ratio)
    val_size = round(n * val_ratio)
    test_size = round(n * test_ratio)

    if((train_size + val_size + test_size) != n):
        val_size = val_size + (n - train_size - val_size - test_size)


    train_data = pd.concat([train_data, dataset.sample(n=train_size)])
    remaining_data = dataset[~dataset.index.isin(train_data.index)]

    val_data = pd.concat([val_data, remaining_data.sample(n=val_size)])
    remaining_data = remaining_data[~remaining_data.index.isin(val_data.index)]
    test_data = pd.concat([test_data, remaining_data.sample(n=test_size)])


input_filename = filename.split('\\')[-1].split('.')[0]
output_file1 = input_filename + '_train.csv'
output_file2 = input_filename + '_val.csv'
output_file3 = input_filename + '_test.csv'
train_data.to_csv(output_file1, index=False, header=None)
val_data.to_csv(output_file2, index=False, header=None)
test_data.to_csv(output_file3, index=False, header=None)