import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

df = pd.read_csv("D:\S\start\code\seeds\\firstdata\double\yongyou9_train.csv", header=None)
train_targets = df.values[:,224]
train_data = df.values[0:800,19:201]
#print(train_targets)
df2 = pd.read_csv("D:\S\start\code\seeds\\firstdata\double\yongyou9_test.csv", header=None)
test_targets = df2.values[:,224]
test_data = df2.values[0:800,19:201]
#print(df2.head(5))
df3 = pd.read_csv("D:\S\start\code\seeds\\firstdata\double\yongyou9_val.csv", header=None)
pre_targets = df3.values[:,224]
pre_data = df3.values[0:800,19:201]

train_tensor_len = len(train_data)
test_tensor_len =  len(test_data)
pre_tensor_len = len(pre_data)
print("训练数据集长度为：{}".format(train_tensor_len))#"训练数据集长度为：{}".format(train_data_size)：格式化字符串。会把{}换成train_data_size的内容
print("验证数据集长度为：{}".format(test_tensor_len))
print("测试数据集长度为：{}".format(pre_tensor_len))


# model = svm.SVC(C=1, kernel='poly') jiayouzhongke 92.8
# model = svm.SVC(C=1, kernel='linear') ning84 94
# model = svm.SVC(C=1, kernel='linear') xiushui121 93
model = svm.SVC(C=1, kernel='linear') 



model.fit(train_data,train_targets)

train_score = model.score(train_data,train_targets)
print("训练集：",train_score)
test_score = model.score(test_data,test_targets)
print("验证集：",test_score)
pre_score = model.score(pre_data,pre_targets)
print("测试集：",pre_score)
