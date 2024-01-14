from random import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

import pandas as pd
import os
import random
import time
import csv
# from Attation import SelfAttention 


# ? 修改位置

# 当我们在每次运行代码时设置相同的随机种子，保证了模型的初始化和训练过程中的随机操作都是相同的，从而确保了每次运行的结果是一致的。这样有助于我们更好地调试代码，对比不同模型或算法的表现，并确保实验的可重现性。否则，如果不设置随机种子，每次运行时随机数的生成都是不同的，导致结果的差异，使得实验不可复现，也难以追踪问题
# 种子数的选择并没有固定的标准，而是需要根据实验的需求和目标来灵活确定。重要的是，在进行实验时，始终保持相同的种子数，以确保实验的可重现性和稳定性
def seed_torch(seed=1029):
    # 设置Python内置random模块的随机种子 这用于在使用random模块的Python函数中进行随机数生成
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
seed_torch()

# 注意力机制
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[1, 2], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[1, 2], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

# 打印控制台输出
# f = open('D:\S\start\code\CodeTest\seedtrain\jiafengyouData\jiafengyou-log.txt','w')

# 创建新的文件用于记录loss和训练次数
# csv_file = "D:\S\start\code\seeds\Binary classification data\data\yongyou1540_laohua192h.csv"
train_losses = []
validation_losses = []
test_losses = []


df = pd.read_csv("jiayouzhongke6_train.csv",  header=None)
train_targets = df.values[:,224]  #? 这里修改为224
train_data = df.values[0:600,19:200]  # 扫描的数据中的行 列
#print(train_targets)
df2 = pd.read_csv("jiayouzhongke6_test.csv",  header=None)
test_targets = df2.values[:,224]  #? 这里修改为224
test_data = df2.values[0:100,19:200]
#print(df2.head(5))

df3 = pd.read_csv("jiayouzhongke6_val.csv", header=None)
pre_targets = df3.values[:,224]  #? 这里修改为224
pre_data = df3.values[0:100,19:200]  


train_data_size = len(train_data)
test_data_size = len(test_data)
pre_data_size = len(pre_data)

print("训练数据集长度为：{}".format(train_data_size))#"训练数据集长度为：{}".format(train_data_size)：格式化字符串。会把{}换成train_data_size的内容
print("验证数据集长度为：{}".format(test_data_size))
print("测试预测集长度为：{}".format(pre_data_size))

# train_data = train_data[:, :181]
train_data = train_data.reshape([-1,1,181])

# test_data = test_data[:, :181]
test_data = test_data.reshape([-1,1,181])

# pre_data = pre_data[:, :181]
pre_data = pre_data.reshape([-1,1,181])

train_data =torch.tensor(train_data,dtype=torch.float32)
train_targets = torch.tensor(train_targets)

test_data = torch.tensor(test_data,dtype=torch.float32)
test_targets = torch.tensor(test_targets)

pre_data = torch.tensor(pre_data,dtype=torch.float32)
pre_targets = torch.tensor(pre_targets)

train_set = TensorDataset(train_data,train_targets)
test_set = TensorDataset(test_data,test_targets)
pre_set = TensorDataset(pre_data,pre_targets)

# ! 过拟合时修改size和学习率
BATCH_SIZE = 32  #8,16,32 最多64
learning_rate = 0.00001
# 数据集  训练量  是否打乱   num_workers，它表示用于数据加载的子进程数量
DataLoader_train_data = DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True,)
DataLoader_test_data = DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=True,)
DataLoader_pre_data = DataLoader(dataset=pre_set,batch_size=BATCH_SIZE,shuffle=True,)
# class CNN(nn.Module):
#     def __init__(self,):
#         super(CNN, self).__init__()
#         self.BN = nn.BatchNorm1d(1)
#         #self.att = dilateformer_tiny()
#         self.atto = simam_module()
#         #self.mutil = MultiHeadAttention(10,180)
# #         self.att = SelfAttention(180)
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(in_channels=1,out_channels=8,kernel_size=2),
#             # nn.MaxPool1d(2),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),

#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(8,16, kernel_size = 2),
#             nn.MaxPool1d(2),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),

#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(16,32,2),
#             # nn.MaxPool1d(2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),

#         )

# # ! 如果过拟合  删除一层或者两层
#         # self.layer4 = nn.Sequential(
#         #     nn.Conv1d(32,64,2),
#         #     nn.BatchNorm1d(64),
#         #     nn.ReLU(),

#         # )

#         # self.layer5 = nn.Sequential(
#         #     nn.Conv1d(64,128,2),
#         #     nn.MaxPool1d(2),
#         #     nn.BatchNorm1d(128),
#         #     nn.ReLU(),

#         # )

#         self.fc1 = nn.Sequential(
#             # 每次需要
#             nn.Linear(2816,512),
#             nn.Dropout(0.4),
#             nn.BatchNorm1d(512),

#           #  nn.Linear(4096,1024),
#             nn.Linear(512,256),
#             nn.Dropout(0.4),
#             nn.BatchNorm1d(256),

#             nn.Linear(256, 128),
#             nn.Dropout(0.4),
#             nn.BatchNorm1d(128),

#             # nn.Linear(128, 32),
#             # nn.Dropout(0.4),
#             # nn.BatchNorm1d(32),

#             nn.Linear(128, 3),

#         )

#     def forward(self, x):
#        # input = torch.randn(40,4,180) 
#         out = self.BN(x)
#         out = self.atto(out)
#         out = self.layer1(out)
#         #out = self.mutil(out)
# #         out = self.att(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
        
#         # out = self.layer4(out)
#         # out = self.layer5(out)
#         out = out.view(out.size(0),-1)
#         out = self.fc1(out)
#       #  out = self.fc4(out)
#        # out = self.fc3(out)
#         return out
class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        self.BN = nn.BatchNorm1d(1)
        self.atto = simam_module()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(8,16, kernel_size = 2),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(16,32,2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=50, num_layers=1, batch_first=True)

        self.fc1 = nn.Sequential(
            nn.Linear(50, 512),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.BatchNorm1d(128),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        out = self.BN(x)
        out = self.atto(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # reshape output for LSTM
        out = out.transpose(1, 2)  # swap the seq_len and num_channels dimensions
        out, _ = self.lstm(out)

        # only take the final time step output for classification
        out = out[:, -1, :]
        out = self.fc1(out)
        return out

zh = CNN()

if torch.cuda.is_available():
    zh = zh.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 定义优化器
# ! 过拟合时修改weight_decay
optimizer = torch.optim.Adam(zh.parameters(),lr=learning_rate,weight_decay= 1e-5)

# 设置训练网络的一些参数
# 设置训练的次数
total_train_step = 0
# 设置测试的次数
total_test_step = 0
# 设置训练的轮数
# ! 过拟合时修改训练轮数
epoch = 1000

start_time = time.time()
best_acc = 0
best_acc_epo = 0

for i in range(epoch):
    total_train_loss = 0
    total_train_acc = 0
    print("-------第{}轮训练开始-------".format(i+1))
    #训练步骤开始
    zh.train()
    for data in DataLoader_train_data:
        imgs,targets = data
        # print(imgs.shape)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = zh(imgs)
        loss = loss_fn(outputs,targets.long())
        total_train_loss = total_train_loss + loss
        #优化器优化模型
        accuracy = (outputs.argmax(1) == targets).sum()  # 详情见tips_1.py
        total_train_acc = total_train_acc + accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(imgs)

        total_train_step = total_train_step + 1

       # if total_train_step % 100 ==0:
           # end_time = time.time()
            #print("Runtime:{}".format(end_time-start_time))
       # print("训练次数：{}，Loss：{}".format(total_train_step,loss))
    print("训练集的Loss:{}".format(total_train_loss))
    print("训练集的正确率：{}".format(total_train_acc/train_data_size))
    train_losses.append(float(total_train_loss))

    zh.eval()
    # 验证集
    total_pre_loss = 0
    total_acc1 = 0
    
    with torch.no_grad():
        for data in DataLoader_pre_data:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = zh(imgs)
            loss = loss_fn(outputs, targets.long())
            total_pre_loss = total_pre_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()  # 详情见tips_1.py
            total_acc1 = total_acc1 + accuracy
            pre_acc = total_acc1 / pre_data_size
    print("验证集的Loss:{}".format(total_pre_loss))
    print("验证集的正确率:{}".format(pre_acc))
    validation_losses.append(float(total_pre_loss))
    
    #测试步骤开始
    total_test_loss = 0
    total_acc = 0
    # best_acc = 0
    # best_acc_epo = 0
    with torch.no_grad():
        for data in DataLoader_test_data:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = zh(imgs)
            loss = loss_fn(outputs,targets.long())
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()#详情见tips_1.py
            total_acc = total_acc + accuracy
            test_acc = total_acc/test_data_size
    print("测试集的Loss:{}".format(total_test_loss))
    print("测试集的正确率:{}".format(test_acc))
    test_losses.append(float(total_test_loss))
    total_test_step = total_test_step + 1




    total_test_step = total_test_step + 1

    # 保留每一轮的模型
    # torch.save(zh.state_dict(),"baizhuo333_CNN_method_{}.pth".format(i+1))
    end_time = time.time()
    print("Runtime:{}".format(end_time-start_time))    
    # print("模型已保存")
    if test_acc >= best_acc:
        best_acc = test_acc
        best_acc_epo = i

# 将三个的loss写入csv文件
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
    
#     # 写入标题行
#     writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Test Loss'])
    
#     # 写入每一轮的损失数据
#     for epoch, (train_loss, validation_loss, test_loss) in enumerate(zip(train_losses, validation_losses, test_losses), start=1):
#         writer.writerow([epoch, train_loss, validation_loss, test_loss])
# print(f"损失数据已写入到 {csv_file} 文件中。")
    
print("最优正确率为：{}".format(best_acc))
print("最优正确率所在的轮数为：{}".format(best_acc_epo+1))

