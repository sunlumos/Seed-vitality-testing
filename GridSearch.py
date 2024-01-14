import pandas as pd
import sklearn.model_selection as ms
from sklearn import svm
from sklearn.linear_model import LogisticRegression

#加载数据
df = pd.read_csv("D:\S\start\code\seeds\\firstdata\double\yongyou9.csv", header=None)
train_targets = df.values[:,224]
train_data = df.values[0:800,19:201]
#设置网格寻优的参数
params = [{'kernel':['linear'], 'C':[1, 10, 100, 1000]},
    {'kernel':['poly'], 'C':[1], 'degree':[2, 3]},
    {'kernel':['rbf'], 'C':[1,10,100,1000], 'gamma':[1, 0.1, 0.01, 0.001]}]

# params = [ {'penalty':['l1','l2'],
#                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
#                   }]

#实例化模型
model = ms.GridSearchCV(svm.SVC(), params,)
# model = ms.GridSearchCV(LogisticRegression(), params, cv=5)

#训练模型
model.fit(train_data, train_targets)
#把上面第12行设置的参数，以排列组合的方式放入模型中run
for p, s in zip(model.cv_results_['params'],
        model.cv_results_['mean_test_score']):
    print(p, s)

# 获取得分最优的的超参数信息
print("得分最优的的超参数信息为:{}".format(model.best_params_))
# 获取最优得分
print("最高准确率为:{}".format(model.best_score_))
# 获取最优模型的信息
print("最优模型的信息为:{}".format(model.best_estimator_))