#!/usr/bin/env python
# coding: utf-8
# ! 基于贝叶斯优化的CNN尝试
"""
作者：胖哥
微信公众号：胖哥真不错
微信号: zy10178083


为了防止大家在运行项目时报错(项目都是运行好的，报错基本都是版本不一致 导致的)，
胖哥把项目中用到的库文件版本在这里说明：

pandas == 1.1.5
matplotlib == 3.3.4
seaborn == 0.11.1
scikit-learn == 0.24.1
tensorflow == 2.4.1
Keras == 2.4.3

"""

# 导入第三方库
from bayes_opt import BayesianOptimization  # 贝叶斯优化库
from sklearn.model_selection import train_test_split  # 导入数据集拆分工具
import numpy as np  # 数据计算库
import warnings  # 告警库
import matplotlib.pyplot as plt  # 数据可视化库
import seaborn as sns  # 高级数据可视化库
import pandas as pd  # 数据处理库
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # 模型评估方法
from keras.models import Sequential  # 导入序贯模型
from keras.utils import plot_model  # 导入 模型绘图工具
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
import keras.layers as layers  # 导入 层 工具包
import keras.backend as K  # 导入 后端 工具包

warnings.filterwarnings(action='ignore')  # 忽略告警


# 贝叶斯目标函数优化卷积神经网络分类模型
def bayesopt_objective_cnn(filters, units, epochs):
    cnn_model = Sequential()  # 序贯模型
    cnn_model.add(Conv1D(filters=int(filters), kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu', padding='valid'))  # 1维卷积层
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))  # 1维最大池化层
    cnn_model.add(Flatten())  # 展平层
    cnn_model.add(Dense(units=int(units), activation='relu'))  # 全连接层
    cnn_model.add(Dense(1, activation='sigmoid'))  # 输出层
    cnn_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['acc'])  # 编译
    cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(epochs), batch_size=64)  # 拟合
    score = cnn_model.evaluate(X_test, y_test, batch_size=128)  # 模型评估

    return score[1]  # 返回验证集分数


# 贝叶斯优化器
def param_bayes_opt_cnn(init_points, n_iter):
    opt = BayesianOptimization(bayesopt_objective_cnn
                               , param_grid_simple
                               , random_state=7)  # 建立贝叶斯优化器对象

    # 使用优化器
    opt.maximize(init_points=init_points  # 抽取多少个初始观测值
                 , n_iter=n_iter  # 总共观测/迭代次数
                 )

    # 返回优化结果
    params_best = opt.max['params']  # 返回最佳参数
    score_best = opt.max['target']  # 返回最佳分数

    return params_best, score_best  # 返回最优参数和分值


# 自定义验证函数，返回bayes_opt最优参数的
def bayes_opt_validation_cnn(params_best):
    cnn_model = Sequential()  # 序贯模型
    cnn_model.add(Conv1D(filters=int(params_best['filters']), kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu'))  # 1维卷积层
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))  # 1维最大池化层
    cnn_model.add(Flatten())  # 展平层
    cnn_model.add(Dense(units=int(params_best['units'])))  # 全连接层
    cnn_model.add(Dense(1, activation='sigmoid'))  # 输出层
    cnn_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['acc'])  # 编译
    cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(params_best['epochs']),
                 batch_size=64)  # 拟合

    score = cnn_model.evaluate(X_test, y_test, batch_size=128)  # 模型评估

    return score[1]  # 返回验证集分数


# 定义主函数
if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv('D:\S\start\code\seeds\\firstdata\double\\ning84.csv')

    # 用Pandas工具查看数据
    print(df.head())

    # 查看数据集摘要
    print(df.info())

    # 数据描述性统计分析
    print(df.describe())

    #  y变量柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # kind='bar' 绘制柱状图
    df['y'].value_counts().plot(kind='bar')
    plt.xlabel("y变量")  # 设置x轴名称
    plt.ylabel("数量")  # 设置y轴名称
    plt.title('y变量柱状图')  # 设置标题名称
    plt.show()  # 展示图片

    # y=1样本x1变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_tmp = df.loc[df['y'] == 1, 'x1']  # 过滤出y=1的样本
    # 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')
    plt.xlabel('x1')  # 设置x轴名称
    plt.ylabel('数量')  # 设置y轴名称
    plt.title('y=1样本x1变量分布直方图')  # 设置标题名称
    plt.show()  # 展示图片

    # 数据的相关性分析

    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    plt.title('相关性分析热力图')  # 设置标题名称
    plt.show()  # 展示图片

    # 提取特征变量和标签变量
    y = df['y']
    X = df.drop('y', axis=1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)  # 增加维度

    print('***********************查看训练集的形状**************************')
    print(X_train.shape)  # 查看训练集的形状

    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)  # 增加维度
    print('***********************查看测试集的形状**************************')
    print(X_test.shape)  # 查看测试集的形状

    param_grid_simple = {'filters': (2.0, 5.0),  # 过滤器数量
                         'units': (20.0, 100.0)  # 隐含层神经元数量
        , 'epochs': (50.0, 100.0)  # 迭代次数
                         }

    params_best, score_best = param_bayes_opt_cnn(10, 10)  # 调用贝叶斯优化器

    print('最优参数组合:  ', 'filters的参数值为：', int(params_best['filters']), 'epochs的参数值为：', int(params_best['epochs']),
          '  units的参数值为：',
          int(params_best['units']))  # 打印最优参数组合
    print('最优分数：  ', abs(score_best))  # 打印最优参数评分
    validation_score = bayes_opt_validation_cnn(params_best)  # 参数组合验证
    print('验证集准确率：  ', abs(validation_score))  # 验证集分数

    # 最优参数构建模型
    cnn_model = Sequential()  # 序贯模型
    cnn_model.add(Conv1D(filters=int(params_best['filters']), kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu'))  # 1维卷积层
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))  # 1维最大池化层
    cnn_model.add(Flatten())  # 展平层
    cnn_model.add(Dense(int(params_best['units']), activation='relu'))  # 全连接层
    cnn_model.add(Dense(1, activation='sigmoid'))  # 输出层
    cnn_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['acc'])  # 编译
    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(params_best['epochs']),
                           batch_size=64)  # 拟合
    print('*************************输出模型摘要信息*******************************')
    print(cnn_model.summary())  # 输出模型摘要信息

    plot_model(cnn_model, to_file='model.png', show_shapes=True)  # 保存模型结构信息


    # 定义绘图函数：损失曲线图和准确率曲线图
    def show_history(history):
        loss = history.history['loss']  # 获取损失
        val_loss = history.history['val_loss']  # 测试集损失
        epochs = range(1, len(loss) + 1)  # 迭代次数
        plt.figure(figsize=(12, 4))  # 设置图片大小
        plt.subplot(1, 2, 1)  # 增加子图
        plt.plot(epochs, loss, 'r', label='Training loss')  # 绘制曲线图
        plt.plot(epochs, val_loss, 'b', label='Test loss')  # 绘制曲线图
        plt.title('Training and Test loss')  # 设置标题名称
        plt.xlabel('Epochs')  # 设置x轴名称
        plt.ylabel('Loss')  # 设置y轴名称
        plt.legend()  # 添加图例
        acc = history.history['acc']  # 获取准确率
        val_acc = history.history['val_acc']  # 获取测试集准确率
        plt.subplot(1, 2, 2)  # 增加子图
        plt.plot(epochs, acc, 'r', label='Training acc')  # 绘制曲线图
        plt.plot(epochs, val_acc, 'b', label='Test acc')  # 绘制曲线图
        plt.title('Training and Test accuracy')  # 设置标题名称
        plt.xlabel('Epochs')  # 设置x轴名称
        plt.ylabel('Accuracy')  # 设置y轴名称
        plt.legend()  # 添加图例
        plt.show()  # 显示图片


    show_history(history)  # 调用绘图函数

    y_pred = cnn_model.predict(X_test, batch_size=10)  # 预测
    y_pred = np.round(y_pred)  # 转化为类别

    print('----------------模型评估-----------------')
    # 模型评估
    print('**************************输出测试集的模型评估指标结果*******************************')

    print('cnn_model分类模型-最优参数-准确率分值: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    print("cnn_model分类模型-最优参数-查准率 :", round(precision_score(y_test, y_pred), 4))
    print("cnn_model分类模型-最优参数-召回率 :", round(recall_score(y_test, y_pred), 4))
    print("cnn_model分类模型-最优参数-F1分值:", round(f1_score(y_test, y_pred), 4))

    from sklearn.metrics import classification_report  # 导入分类报告工具

    # 分类报告
    print(classification_report(y_test, y_pred))

    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵工具
    import seaborn as sns  # 统计数据可视化

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 构建数据框
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual :0', 'Actual :1'],
                             index=['Predict :0', 'Predict :1'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')  # 热力图展示
    plt.show()  # 展示图片
