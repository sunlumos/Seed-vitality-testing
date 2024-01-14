import pandas as pd
import numpy as np
from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
from tcn import TCN
import torch

print(torch.__version__)
# 指定训练集和测试集的文件路径
train_data_path = 'D:\S\start\code\seeds\\firstdata\double\jiayouzhongke6_train.csv'
test_data_path = 'D:\S\start\code\seeds\\firstdata\double\jiayouzhongke6_test.csv'

# 从CSV文件中读取训练集
train_df = pd.read_csv(train_data_path)
x_train = train_df[train_df.columns[19:200]]
y_train = train_df[train_df.columns[-1]]

# 从CSV文件中读取测试集
test_df = pd.read_csv(test_data_path)
x_test = test_df[test_df.columns[19:200]] 
y_test = test_df[test_df.columns[-1]]

# 数据预处理，转换为NumPy数组并调整形状以适合模型
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# 调整输入数据的形状以适合模型
input_shape = (x_train.shape[1], 1)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

model = keras.models.Sequential()
model.add(TCN(input_shape=input_shape))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, epochs=100, batch_size=32)

# 找到最佳准确率及其对应的轮次
best_epoch = np.argmax(history.history['accuracy'])
best_accuracy = history.history['accuracy'][best_epoch]

print('Best Accuracy: {:.2f}%'.format(best_accuracy * 100))
print('Best Epoch: {}'.format(best_epoch + 1))  # epoch计数从1开始

# 验证模型
model.evaluate(x_test, y_test)