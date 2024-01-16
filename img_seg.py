# 图像分割代码
import cv2

# img_png = cv2.imread(r"test.jpg")
#
# img_png = img_png[50:130,500:540]
# # # img_png = img_png[260:400,350:430]
# #
# cv2.imshow("img_binary", img_png)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# width = 220
# high = 107
# for rows in range(4):
#     for cols in range(5):
#         img_slice = img_png[40 + width * rows :180 + width * rows, 450 - high * cols : 530 - high * cols]
#         cv2.imshow("img_binary", img_slice)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#上下格子之间的平移距离
high =50
#左右之间的平移距离
width =45
#第一个格子的边界范围
boundary_x1,boundary_x2 = 50,130   #纵向 上下方向
boundary_y1,boundary_y2 = 495,540  #横向 左右方向
#行数
rows_num = 6
#列数
cols_num = 11
#读取数据
img_png = cv2.imread(r"D:\S\start\code\1.jpg")
for rows in range(rows_num):
    for cols in range(cols_num):
        img_slice = img_png[boundary_x1 + high * rows :boundary_x2 + high * rows, boundary_y1 - width * cols : boundary_y2 - width * cols]
        cv2.imshow("img_binary", img_slice)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    break
