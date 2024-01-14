import glob
import Auto_getHy

fileName = Auto_getHy.seedName
# oldAged = Auto_getHy.oldAged


csv_list = glob.glob(r'D:\S\start\\code\\getHy\\data\\{}-*.csv'.format(fileName))  # 查看同文件夹下的csv文件数
print(u'共发现%s个CSV文件' % len(csv_list))
print(u'正在处理............')

for i in csv_list:  # 循环读取同文件夹下的csv文件
    fr = open(i, 'rb').read()
    with open(r'D:\S\start\\code\\getHy\\data\\{}.csv'.format(fileName), 'ab') as f:  # 将结果保存为result.csv
        f.write(fr)
print('合并完毕！')

