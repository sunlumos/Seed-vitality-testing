import pandas as pd
from sklearn.model_selection import train_test_split

# ! 划分数据集的代码

fileName = 'xiushui121'#####################

def save(x,name):
    x = pd.DataFrame(x,index=None,columns=None)
    x.to_csv("D:\S\start\code\getHy\splitData\{}-{}".format(fileName,name) + ".csv", index=0, sep=',', header=None)

# CSVdata = pd.read_csv(r"Triple classification data\{}".format(fileName) + ".csv",header=None,)
CSVdata = pd.read_csv(r"D:\S\start\\code\\getHy\\data\\xiushui121.csv",header=None,)

data = CSVdata.values[0:800,0:225] #[行，列] *********这里故意把Label也读到data变量中，这样在save自定义函数中就不用再降数据和标签拼接了
label = CSVdata.values[:,224]

x_train, x_test,  y_train, y_test = train_test_split(data, label, test_size = 3/4, random_state = 7)
save(x_train,'train')
save(x_test,'transitionFilesTest')

CSVdata = pd.read_csv("D:\S\start\code\getHy\splitData\{}-transitionFilesTest".format(fileName) + ".csv",header=None,)
data = CSVdata.values[0:267,0:225] #[行，列] *********这里故意把Label也读到data变量中，这样在save自定义函数中就不用再降数据和标签拼接了
label = CSVdata.values[:,224]
x_train, x_test,  y_train, y_test = train_test_split(data, label, test_size = 1/2, random_state = 7)
save(x_train,'val')
save(x_test,'test')
