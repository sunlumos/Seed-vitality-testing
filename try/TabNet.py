import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt

# 读取训练集和测试集
train_data = pd.read_csv(r'D:/S/start/code/seeds/firstdata/double/jiayouzhongke6_train.csv')
test_data = pd.read_csv(r'D:/S/start/code/seeds/firstdata/double/jiayouzhongke6_test.csv')

# 提取训练集的特征和标签
X_train = train_data.iloc[:, 19:200].values  # 20到200列作为特征
y_train = train_data.iloc[:, -1].values   # 最后一列作为标签

# 提取测试集的特征和标签
X_test = test_data.iloc[:, 19:200].values
y_test = test_data.iloc[:, -1].values

# 数据预处理
# 如果需要对特征进行预处理，如标准化、归一化等，可以在此进行操作

# 模型搭建
clf = TabNetClassifier()

# 训练模型
history = clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'test'],
    max_epochs=11,
    batch_size=64,
    virtual_batch_size=32,
    patience=11
)

# 模型评估
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)

# 绘制损失和准确率曲线
train_loss = history['train']['loss']
train_acc = history['train']['accuracy']
test_loss = history['test']['loss']
test_acc = history['test']['accuracy']



plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

