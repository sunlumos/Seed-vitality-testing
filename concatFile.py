import glob

# ! 把提出来的CSV光谱文件 合并成一个CSV文件的代码

fileName = 'yongyou12'
oldAged = 'weilaohua'

csv_list = glob.glob(r'data\{}-*{}.csv'.format(fileName,oldAged))  # 查看同文件夹下的csv文件数
print(u'共发现%s个CSV文件' % len(csv_list))
print(u'正在处理............')
epoch = 1
for i in csv_list:  # 循环读取同文件夹下的csv文件
    fr = open(i, 'rb').read()
    with open(r'data\{}_{}.csv'.format(fileName,oldAged), 'ab') as f:  # 将结果保存为result.csv
        f.write(fr)
print('合并完毕！')

