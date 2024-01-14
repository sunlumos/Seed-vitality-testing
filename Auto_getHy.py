import os
import cv2
import numpy as np
import csv
from spectral.io import envi

# ! 提取光谱

file_dir = "C:\\PyDaima\\TiQuGuangPu"
saveGrayPhotodir =  "C:\\PyDaima\\TiQuGuangPu\\photo"
listdir = os.listdir(file_dir)
filePath_list = []
previousRoundHdr = None

for epoch in range(0,6):
    s = 'shaonuo9714-{}'.format(epoch+1)
    for filewalks in os.walk(file_dir):
        for files in filewalks[2]:
            # print('true files',files)
            if s in files:
                # print(s, ' is in', os.path.join(filewalks[0], files))
                filePath_list.append(os.path.join(filewalks[0], files))
    # print(filePath_list[0])

    for i in filePath_list:
        if os.path.splitext(i)[1] == ".hdr" :  # 筛选文件
            hdrFile = i
            print(hdrFile)
        if os.path.splitext(i)[1] == ".dat":  # 筛选文件
            datFile = i
            print(datFile)
        if os.path.splitext(i)[1] == ".png":  # 筛选文件
            rgbFile = i
            print(rgbFile)

    if hdrFile == previousRoundHdr:
        print("光谱已经全部提取完毕！")
        break
    # 读入图片
    # numpy transpose 转换
    img0 = envi.open(r'{}'.format(hdrFile),
                     r'{}'.format(datFile))

    img0 = img0.load()
    img0 = img0[200:900, 0:640]
    img = cv2.imread("{}".format(rgbFile))
    img = img[200:900, 0:640]  # 裁剪坐标为[y0:y1, x0:x1]
    print("The size of picture：{}".format(img.shape))  # 高度、宽度、通道数
    # img2 = img #拷贝一份原图
    # 中值滤波，去噪
    img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("img",img)

    # 阈值分割得到二值化图片
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 膨胀操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    bin_clo = cv2.dilate(binary, kernel2, iterations=2)

    # 输入：image : 是要处理的图片，官方文档要求是8位单通道的图像。
    # connectivity : 可以选择是4连通还是8连通。

    # 输出：retval : 返回值是连通区域的数量。
    # labels : labels是一个与image一样大小的矩形（labels.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始。
    # stats ：stats会包含5个参数分别为x,y,h,w,s。分别对应每一个连通区域的外接矩形的起始坐标x,y；外接矩形的wide,height；s其实不是外接矩形的面积，实践证明是labels对应的连通区域的像素个数。
    # centroids : 返回的是连通区域的质心。

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=bin_clo,
                                                                            connectivity=8)  # 4连通指的是上下左右，8连通指的是上下左右+左上、右上、右下、左下。
    centroids = centroids.astype(np.int32)  # 中心点坐标取整
    # 查看各个返回值
    # 连通域数量
    print('num_labels = ', num_labels)  # 101是因为背景算作一个大的连通图
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    # print('stats = ',stats)
    # 连通域的中心点
    # print('centroids = ',centroids)

    # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    # print('labels = ',labels)
    np.savetxt('data/labels.csv', labels, fmt='%.4f', delimiter=',')

    # # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    print(output.shape)  # (400, 640, 3)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 255)  # output的第0个通道的全部坐标点，B通道
        output[:, :, 1][mask] = np.random.randint(0, 255)  # output的第1个通道的全部坐标点，G通道
        output[:, :, 2][mask] = np.random.randint(0, 255)  # output的第2个通道的全部坐标点，R通道

    # 排序、标记
    list1 = []  # 把中心点坐标存入列表
    for k, i in enumerate(centroids):  # 用中心点排序
        if k >= 1:
            ll = tuple(i[:2])  # 只读取前面两列数据
            # print(ll)
            list1.append(ll)

    step = 10
    zipped_10 = [list1[i:i + step] for i in range(0, 99, step)]
    # print("zipped_10={}".format(zipped_10))
    zimu = []  # 存放A1-J10 标定序号的列表
    for i in range(65, 75, ):
        # for j in range(10,0,-1):
        for j in range(1, 11, ):
            zimu.append(chr(i) + str(j))

    z = 0
    list_sortted = []  # 存放每十个一排序的坐标，按X从大到小排序
    for x in zipped_10:
        x = sorted(x, reverse=True)  # 每十个排序一次，按照x坐标从大到小排序
        list_sortted.append(x)
        # print(x)
        for j in range(10):
            # print(x[j])
            cv2.putText(output, zimu[z], x[j], cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, )  # 在照片上标号
            # print(chr(i)+str(j)+str(x[z]))
            z = z + 1
    # print(list_sortted)
    fi = open("C:\\PyDaima\\TiQuGuangPu\\data\\labels.csv", 'r', encoding="utf-8")
    reader = csv.reader(fi)
    rows = [row for row in reader]
    list_x = []  # x坐标
    list_y = []  # y坐标
    key = []  # key值
    for k, i in enumerate(rows):
        for l, j in enumerate(i):
            if int(float(j)) != 0:
                # print(l+1,k+1,j)
                list_x.append(l)
                list_y.append(k)
                key.append(j)

    zipped_xy = zip(list_x, list_y)  # 坐标的形式，例如(193, 41)
    zipped_xy_key = list(zip(zipped_xy, key))  # 坐标+其对应的像素标号的形式，例如((379, 58), '3')

    paixuhaodezuobiao = []
    for sor in list_sortted:  # 从全部排好序的坐标点中拿每10个排好序的坐标点，例如拿出A1-A10zipped_xy_key = {list: 14477} [((230, 128), '1.0000'), ((275, 128), '2.0000'), ((276, 128), '2.0000'), ((277, 128), '2.0000'), ((204, 129), '3.0000'), ((205, 129), '3.0000'), ((206, 129), '3.0000'), ((207, 129), '3.0000'), ((208, 129), '3.0000'), ((229, 129), '1.0000'), ((230, 129), '1… View
        # print(sor)
        for z in sor:  # 从每10个排好序的坐标点中拿一个坐标点，例如拿出A1
            paixuhaodezuobiao.append(z)
    print(paixuhaodezuobiao)

    zuobiaoduiyingdeKey = []
    for zuobiao in paixuhaodezuobiao:
        for j in zipped_xy_key:
            if zuobiao == j[0]:
                zuobiaoduiyingdeKey.append(j[1])
    print(zuobiaoduiyingdeKey)
    xiangsu = []
    i = 1
    for lala in zuobiaoduiyingdeKey:
        sum_px = 0
        count = 0
        for haha in zipped_xy_key:
            if haha[1] == lala:
                sum_px = sum_px + img0[haha[0][1], haha[0][
                    0]]  # 在使用image(x1, x2)来访问图像中点的值的时候，x1并不是图片中对应点的x轴坐标，而是图片中对应点的y坐标。因此其访问的结果其实是访问image图像中的Point(x2, x1)点，即与image.at(Point(x2, x1))效果相同。
                count += 1
        xiangsu.append(sum_px / count)
        print("第{}个连通域：该连通域内共有{}个像素点".format(i, count))
        i = i + 1

    np.savetxt('{}\\data\\{}.csv'.format(file_dir,s), xiangsu, fmt='%.4f', delimiter=',')
    cv2.imwrite('{}\\{}.png'.format(saveGrayPhotodir,s), output);
    filePath_list = []
    previousRoundHdr = hdrFile


