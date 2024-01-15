# 用于种子自动检测机器的自动提取光谱代码
import cv2
import numpy as np
import csv
# 读入图片
from spectral.io import envi
import matplotlib.pyplot as plt
#numpy transpose 转换



#1.获取高光谱图片数据
img0 = envi.open(r"F:\dataset\RAW\daoguangpu-xigaufanqie\xiguafanqie(1)-2023-7-24\caihongguazhibao_2023-07-23_11-56-01\capture\REFLECTANCE_caihongguazhibao_2023-07-23_11-56-01.hdr",
                r"F:\dataset\RAW\daoguangpu-xigaufanqie\xiguafanqie(1)-2023-7-24\caihongguazhibao_2023-07-23_11-56-01\capture\REFLECTANCE_caihongguazhibao_2023-07-23_11-56-01.dat")
# print("img0.shape",img0)
# #1.1获取波段范围
# band_wavelengths = img0.bands.centers
# print("wavelengths: ",band_wavelengths)
# print("wavelengths len: ",len(band_wavelengths))
# print("type ", np.array(band_waveengths))
# np.savetxt('F:\dataset\\1Dshuju-fanqie\\chuweis350_2023-07-23_10-34-48.csv',np.array([band_wavelengths]),fmt='%.4f',delimiter=',')
#
# img0 = img0.load()
# img0 = img0[270:655, 0:640]
# print(img0.shape)
# print("img0",img0[0,0].shape)
img_png = cv2.imread(r"C:\Users\bruce\Desktop\test.png")
# cv2.imshow("img_png",img_png)
# cv2.waitKey(0)
img_png= img_png[:, :]# 裁剪坐标为[y0:y1, x0:x1]
# print("The size of picture：{}".format(img_png.shape))#高度、宽度、通道数
#img_png2 = img_png #拷贝一份原图

#2.对于高光谱图片滤波去噪
# 中值滤波，去噪
img = cv2.medianBlur(img_png, 3)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('original', img_gray)
# cv2.imshow("img",img)


#3.阈值分割得到二值化图片
ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("img_binary",img_binary)
cv2.waitKey(0)


#4.膨胀操作
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
img_dilated = cv2.dilate(img_binary, kernel2, iterations=2)


#5.连通域分析，就是几个ROI（感兴趣区域）
# 输入：image : 是要处理的图片，官方文档要求是8位单通道的图像。4连通指的是上下左右，8连通指的是上下左右+左上、右上、右下、左下。
#      connectivity : 可以选择是4连通还是8连通。

# 输出：retval : 返回值是连通区域的数量。
#      labels : labels是一个与image一样大小的矩形（labels.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始。
#      stats : stats会包含5个参数分别为x,y,h,w,s。分别对应每一个连通区域的外接矩形的起始坐标x,y；外接矩形的wide,height；s其实不是外接矩形的面积，实践证明是labels对应的连通区域的像素个数。
#      centroids : 返回的是连通区域的质心。
#5.1连通域信息的获取
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=img_dilated, connectivity=8)
centroids = centroids.astype(np.int32)#中心点坐标取整

#5.2查看各个返回值
# 连通域数量  101是因为背景算作一个大的连通图
print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
# print('stats = ',stats)
# 连通域的中心点
# print('centroids = ',centroids)
# # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
# print('labels = ',labels)
# 创建一个 colormap（颜色映射）来显示不同的连通组件
# cmap = plt.get_cmap('nipy_spectral')
#
# # 将 labels 矩阵可视化为彩色图像
# labels_colored = cmap(labels)
# plt.imshow(labels_colored)
# # 显示图像
# plt.colorbar()
# plt.show()
# print(labels.shape)
#np.savetxt('data\labels.csv',labels,fmt='%.4f',delimiter=',')
#
#6.不同的连通域赋予不同的颜色
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
# print(output.shape)  #(400, 640, 3)
# for i in range(1, num_labels):
#     mask = labels == i
#     output[:, :, 0][mask] = np.random.randint(0, 255)#output的第0个通道的全部坐标点，B通道
#     output[:, :, 1][mask] = np.random.randint(0, 255)#output的第1个通道的全部坐标点，G通道
#     output[:, :, 2][mask] = np.random.randint(0, 255)#output的第2个通道的全部坐标点，R通道

#7 对每一个ROI（感兴趣区域） 进行编号(A1-J10)
#7.1获取中心坐标
list_centroids= centroids[1:]

#7.2 每10个一组，进行标号(A1-J10)生成
step = 11
zipped_10 = [list_centroids[i:i+step] for i in range(0,num_labels-1,step)]
# print(zipped_10)
#print("zipped_10={}".format(zipped_10))
# zimu = []#存放A1-J10 标定序号的列表
# for i in range(65,77,):#A-67 J-74
#     for j in range(1,12,):
#         zimu.append(chr(i)+str(j))
#print(zimu)

#7.3 按A1-J10 的坐标排序好，图片左上角A10 右上角A1，右下角J1，左下角J10
z = 0
list_sorted = []#存放每十个一排序的坐标，按X从大到小排序
for x in zipped_10:
    # 使用numpy.argsort()获取按第一个坐标排序后的索引
    sorted_indices = np.argsort(x[:, 0])
    # 根据排序后的索引对数据进行重新排序，每组（10个）并按x轴坐标从大到小排序
    sorted_data = x[sorted_indices][::-1]
    # print("sorted: ",sorted_data)
    list_sorted.append(sorted_data)
    # print(sorted_data.shape[0])
    # print("sorted_data",sorted_data.shape)
    # for j in range(sorted_data.shape[0]):
    #     # 在照片上标号
    #     cv2.putText(output,zimu[z],tuple(sorted_data[j]),cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1,)
    #     # print(chr(i)+str(j)+str(x[z]))
    #     z = z + 1
#print("list_sorted: ",list_sorted)


#8.计算 像素对应的值
hyperspectral_data =[]
#遍历每组
for group in list_sorted:
    #提取每组的坐标
    for element in group:
        sum_px = 0
        count = 0
        # print(element)
        # print(type(element))
        # print(labels[element[1]][element[0]])
        # 使用numpy.where()函数找到值为对应坐标的编号值的元素坐标
        rows, cols = np.where(labels == labels[element[1]][element[0]])
        for index in range(len(rows)):
            #print(rows[index],cols[index])
            sum_px+=img0[rows[index],cols[index]]
            count+=1
        #print("count",count)
        hyperspectral_data.append(sum_px/count)
# print(labels[128][203])
# with open("F:\dataset\\1Dshuju-fanqie\\chuweis350_2023-07-23_10-34-48.csv", mode="a") as file:
np.savetxt(r'F:\dataset\RAW\1Dshuju-xigua\xigua3\mingyu1_20', hyperspectral_data, fmt='%.4f', delimiter=',')
# # print(hyperspectral_data)

# cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

