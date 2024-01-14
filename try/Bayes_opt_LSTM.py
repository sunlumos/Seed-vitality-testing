#!/usr/bin/env python
# coding: utf-8
# ! Python实现哈里斯鹰优化算法(HHO)优化循环神经网络分类模型(LSTM分类算法)
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

"""

# 导入第三方库
import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential  # 导入序贯模型
from keras.layers import Dense  # 导入全连接层
from keras.layers import LSTM  # 导入LSTM层
from keras.utils import plot_model  # 导入 模型绘图工具
import keras.layers as layers  # 导入 层 工具包
import keras.backend as K  # 导入 后端 工具包
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # 模型评估方法



# 定义初始化位置函数
def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')  # 位置初始化为0
    for i in range(N):  # 循环
        for d in range(dim):  # 循环
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()  # 位置随机初始化

    return X  # 返回位置数据


# 定义转换函数
def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')  # 位置初始化为0
    for i in range(N):  # 循环
        for d in range(dim):  # 循环
            if X[i, d] > thres:  # 判断
                Xbin[i, d] = 1  # 赋值
            else:
                Xbin[i, d] = 0  # 赋值

    return Xbin  # 返回数据


# 定义边界处理函数
def boundary(x, lb, ub):
    if x < lb:  # 小于最小值
        x = lb  # 赋值最小值
    if x > ub:  # 大于最大值
        x = ub  # 赋值最大值

    return x  # 返回位置数据


# 定义莱维飞行函数
def levy_distribution(beta, dim):
    # Sigma计算赋值
    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)  # 计算
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)  # 计算
    sigma = (nume / deno) ** (1 / beta)  # Sigma赋值
    # Parameter u & v
    u = np.random.randn(dim) * sigma  # u参数随机赋值
    v = np.random.randn(dim)  # v参数随机赋值
    # 计算步骤
    step = u / abs(v) ** (1 / beta)  # 计算
    LF = 0.01 * step  # LF赋值

    return LF  # 返回数据


# 定义错误率计算函数
def error_rate(X_train, y_train, X_test, y_test, x, opts):
    if abs(x[0]) > 0:  # 判断取值
        units = int(abs(x[0])) * 10  # 赋值
    else:
        units = int(abs(x[0])) + 16  # 赋值

    if abs(x[1]) > 0:  # 判断取值
        epochs = int(abs(x[1])) * 10  # 赋值
    else:
        epochs = int(abs(x[1])) + 10  # 赋值

    # 建支持LSTM模型并训练
    lstm = Sequential()  # 序贯模型
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # LSTM层
    lstm.add(LSTM(units=units))  # LSTM层
    lstm.add(Dense(10, activation='relu'))  # 全连接层
    lstm.add(Dense(1, activation='sigmoid'))  # 输出层
    lstm.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['acc'])  # 编译
    lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)  # 拟合
    score = lstm.evaluate(X_test, y_test, batch_size=128)  # 模型评估

    # 使错误率降到最低
    fitness_value = (1 - float(score[1]))  # 错误率 赋值 适应度函数值

    return fitness_value  # 返回适应度


# 定义目标函数
def Fun(X_train, y_train, X_test, y_test, x, opts):
    # 参数
    alpha = 0.99  # 赋值
    beta = 1 - alpha  # 赋值
    # 原始特征数
    max_feat = len(x)
    # 选择特征数
    num_feat = np.sum(x == 1)
    # 无特征选择判断
    if num_feat == 0:  # 判断
        cost = 1  # 赋值
    else:
        # 调用错误率计算函数
        error = error_rate(X_train, y_train, X_test, y_test, x, opts)
        # 目标函数计算
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost  # 返回数据


# 定义哈里斯鹰优化算法主函数
def jfs(X_train, y_train, X_test, y_test, opts):
    # 参数
    ub = 1  # 上限
    lb = 0  # 下限
    thres = 0.5  # 阀值
    beta = 1.5  # levy 参数

    N = opts['N']  # 种群数量
    max_iter = opts['T']  # 最大迭代次数
    if 'beta' in opts:  # 判断
        beta = opts['beta']  # 赋值

    # 维度
    dim = np.size(X_train, 1)  # 获取维度
    if np.size(lb) == 1:  # 判断
        ub = ub * np.ones([1, dim], dtype='float')  # 初始化上限为1
        lb = lb * np.ones([1, dim], dtype='float')  # 初始化下限为1

    # 调用位置初始化函数
    X = init_position(lb, ub, N, dim)

    fit = np.zeros([N, 1], dtype='float')  # 适应度初始化为0
    Xrb = np.zeros([1, dim], dtype='float')  # 猎物位置初始化为1
    fitR = float('inf')  # 初始化为无穷

    curve = np.zeros([1, max_iter], dtype='float')  # 适应度初始化为0
    t = 0  # 赋值

    while t < max_iter:  # 循环
        # 调用转换函数
        Xbin = binary_conversion(X, thres, N, dim)

        # 计算适应度
        for i in range(N):  # 循环
            fit[i, 0] = Fun(X_train, y_train, X_test, y_test, Xbin[i, :], opts)  # 调用目标函数
            if fit[i, 0] < fitR:  # 判断
                Xrb[0, :] = X[i, :]  # 猎物位置赋值
                fitR = fit[i, 0]  # 适应度赋值

        # 存储结果
        curve[0, t] = fitR.copy()  # 复制
        print("*********************************", "当前迭代次数: ", t + 1, "***************************************")
        print("最好的适应度数值: ", curve[0, t])
        t += 1

        # 平均位置
        X_mu = np.zeros([1, dim], dtype='float')  # 初始化为0
        X_mu[0, :] = np.mean(X, axis=0)  # 计算平均位置

        for i in range(N):  # 循环
            E0 = -1 + 2 * rand()  # 猎物的初始能量  [-1,1] 之间的随机数
            E = 2 * E0 * (1 - (t / max_iter))  # 逃逸能量
            # 当|E|≥1 时进入搜索阶段
            if abs(E) >= 1:
                q = rand()  # 生成随机数 [0,1]
                if q >= 0.5:  # 判断
                    k = np.random.randint(low=0, high=N)  # 生成随机整数  个体
                    r1 = rand()  # [0,1]之间的随机数
                    r2 = rand()  # [0,1]之间的随机数
                    for d in range(dim):  # 循环
                        X[i, d] = X[k, d] - r1 * abs(X[k, d] - 2 * r2 * X[i, d])  # 更新位置
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  # 边界处理

                elif q < 0.5:  # 判断
                    r3 = rand()  # [0,1]之间的随机数
                    r4 = rand()  # [0,1]之间的随机数
                    for d in range(dim):  # 循环
                        X[i, d] = (Xrb[0, d] - X_mu[0, d]) - r3 * (lb[0, d] + r4 * (ub[0, d] - lb[0, d]))  # 更新位置
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  # 边界处理


            elif abs(E) < 1:  # 开发阶段
                J = 2 * (1 - rand())  # 生成随机数
                r = rand()  # 生成随机数
                # 软围攻策略进行位置更新
                if r >= 0.5 and abs(E) >= 0.5:
                    for d in range(dim):  # 循环
                        DX = Xrb[0, d] - X[i, d]  # 猎物位置与个体当前位置的差值
                        X[i, d] = DX - E * abs(J * Xrb[0, d] - X[i, d])  # 更新位置
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  # 边界处理

                # 硬围攻策略进行位置更新
                elif r >= 0.5 and abs(E) < 0.5:
                    for d in range(dim):  # 循环
                        DX = Xrb[0, d] - X[i, d]  # 猎物位置与个体当前位置的差值
                        X[i, d] = Xrb[0, d] - E * abs(DX)  # 更新位置
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  # 边界处理

                # 渐近式快速俯冲的软包围策略进行位置更新
                elif r < 0.5 and abs(E) >= 0.5:
                    LF = levy_distribution(beta, dim)  # 莱维飞行
                    Y = np.zeros([1, dim], dtype='float')  # 初始化为0
                    Z = np.zeros([1, dim], dtype='float')  # 初始化为0

                    for d in range(dim):  # 循环

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X[i, d])  # 更新位置

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])  # 边界处理

                    for d in range(dim):  # 循环

                        Z[0, d] = Y[0, d] + rand() * LF[d]  # 更新位置

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])  # 边界处理

                        # 调用转换函数
                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    # 适应度计算
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)
                    # 根据适应度进行判断
                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY  # 赋值
                        X[i, :] = Y[0, :]  # 赋值
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ  # 赋值
                        X[i, :] = Z[0, :]  # 赋值

                # 带有莱维飞行的硬围攻策略进行位置更新
                elif r < 0.5 and abs(E) < 0.5:
                    # Levy distribution (9)
                    LF = levy_distribution(beta, dim)  # 莱维飞行
                    Y = np.zeros([1, dim], dtype='float')  # 初始化为0
                    Z = np.zeros([1, dim], dtype='float')  # 初始化为0

                    for d in range(dim):  # 循环

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X_mu[0, d])  # 更新位置

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])  # 边界处理

                    for d in range(dim):  # 循环

                        Z[0, d] = Y[0, d] + rand() * LF[d]  # 更新位置

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])  # 边界处理

                        # 调用转换函数
                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    # 适应度计算
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)
                    # 根据适应度进行判断
                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY  # 赋值
                        X[i, :] = Y[0, :]  # 赋值
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ  # 赋值
                        X[i, :] = Z[0, :]  # 赋值

    return X  # 返回数据


if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv('D:\S\start\code\seeds\\firstdata\jiayouzhongke6.csv')

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
    plt.xlabel("y变量")
    plt.ylabel("数量")
    plt.title('y变量柱状图')
    plt.show()

    # y=1样本x1变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_tmp = df.loc[df['y'] == 1, 'x1']  # 过滤出y=1的样本
    # 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')
    plt.xlabel('x1')
    plt.ylabel('数量')
    plt.title('y=1样本x1变量分布直方图')
    plt.show()

    # 数据的相关性分析
    import seaborn as sns

    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    plt.title('相关性分析热力图')
    plt.show()

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

    # 参数初始化
    N = 10  # 种群数量
    T = 2  # 最大迭代次数

    opts = {'N': N, 'T': T}

    # 调用哈里斯鹰优化算法主函数
    fmdl = jfs(X_train, y_train, X_test, y_test, opts)

    if abs(fmdl[0][0]) > 0:  # 判断
        best_units = int(abs(fmdl[0][0])) * 10 + 48 # 赋值
    else:
        best_units = int(abs(fmdl[0][0])) + 48  # 赋值

    if abs(fmdl[0][1]) > 0:  # 判断
        best_epochs = int(abs(fmdl[0][1])) * 10 + 60  # 赋值
    else:
        best_epochs = (int(abs(fmdl[0][1])) + 100)  # 赋值

    print('----------------HHO哈里斯鹰优化算法优化LSTM模型-最优结果展示-----------------')
    print("The best units is " + str(abs(best_units)))
    print("The best epochs is " + str(abs(best_epochs)))

    # 应用优化后的最优参数值构建LSTM分类模型
    lstm = Sequential()  # 序贯模型
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # LSTM层
    lstm.add(LSTM(units=best_units))  # LSTM层
    lstm.add(Dense(10, activation='relu'))  # 全连接层
    lstm.add(Dense(1, activation='sigmoid'))  # 输出层
    lstm.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['acc'])  # 编译
    history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_epochs, batch_size=64)  # 拟合
    print('*************************输出模型摘要信息*******************************')
    print(lstm.summary())  # 输出模型摘要信息

    plot_model(lstm, to_file='model.png', show_shapes=True)  # 保存模型结构信息


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

    y_pred = lstm.predict(X_test, batch_size=10)  # 预测
    y_pred = np.round(y_pred)  # 转化为类别

    print('----------------模型评估-----------------')
    # 模型评估
    print('**************************输出测试集的模型评估指标结果*******************************')

    print('LSTM分类模型-最优参数-准确率分值: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    print("LSTM分类模型-最优参数-查准率 :", round(precision_score(y_test, y_pred), 4))
    print("LSTM分类模型-最优参数-召回率 :", round(recall_score(y_test, y_pred), 4))
    print("LSTM分类模型-最优参数-F1分值:", round(f1_score(y_test, y_pred), 4))

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
