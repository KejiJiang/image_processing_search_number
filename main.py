import cv2
import numpy as np


# 形态学处理
def Process(img):
    # 高斯平滑，对图像降噪处理，img是原图像对象，（3,3）是设置的高斯平滑的宽与高，0和0分别是x和y上的高斯核标准偏差
    gaussian = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # 中值滤波，gaussian是需要处理的图像，即对高斯平滑处理后的图像进行中值滤波，5是滤波模板的尺寸
    median = cv2.medianBlur(gaussian, 5)
    # Sobel算子，是高斯平滑与微分操作的结合体，所以它的抗噪声能力很好
    # 梯度方向: x
    # 对median进行操作，输出的深度存入CV_8U，8位无符号整数，1，0表示在x方向求一阶导，y方向不求导，即梯度方向为x，
    # ksize=3表示Sobel算子大小为3
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化，170是起始阈值，255是最大阈值，cv2.THRESH_BINARY处理阈值关系，像素值大于170的，则改为255，否则改为0
    # binary是二值化后的图像
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    # 构造核函数,cv2.MORPH_RECT矩形结构元素，所有元素值都是1，(9,1)和(9,7)是矩形元素的大小
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    # 先膨胀再腐蚀，是闭运算，被用来填充前景物体中的小洞，或者前景物体上的小黑点
    # 随后再膨胀，是开运算，用来去除噪音，排除小团的物体
    # 膨胀，对binary进行膨胀处理，使其白色区域（前景）增加，element2是膨胀操作的内核，iteration=1说明膨胀次数为1
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀，对dilation进行腐蚀处理，element1是其操作的内核，iteration=1说明腐蚀次数为1
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 膨胀，对erosion进行膨胀处理，使其白色区域（前景）增加，element2是膨胀操作的内核，iteration=3说明膨胀次数为3
    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    #cv2.imshow('dilation2',dilation2)
    return dilation2


def GetRegion(img):
    regions = []
    # 在二值图像中查找轮廓，img为二值图像
    # cv2.RETR_TREE是轮廓检索方式，建立一个等级树结构的轮廓
    # cv2.CHAIN _APPROX _SIMPLE是轮廓近似方法，压缩垂直、水平、对角方向，只保留端点
    # 返回的contours是轮廓点集，hierarchy是轮廓层次结构
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 返回轮廓点集组成的面积
        area = cv2.contourArea(contour)
        if (area < 2000):
            continue
        # minAreaRect返回一个旋转矩阵，其中包含矩形左上角角点的坐标（x，y），矩形的宽和高（w，h），以及旋转角度
        rect = cv2.minAreaRect(contour)
        # boxPoint返回一个旋转矩阵的四个顶点
        box = cv2.boxPoints(rect)
        # 对四个顶点的坐标转化为整型
        box = np.int0(box)
        # 得到四个顶点所构成长方体的高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 得到宽高比
        ratio = float(width) / float(height)
        # 根据大部分的车牌宽高比比，筛选点集
        if (ratio < 5 and ratio > 1.8):
            # 添加符合要求的顶点集
            regions.append(box)
    return regions


def detect(img):
    # 灰度化，使用颜色空间转换函数cvtColor，img是需要处理的图片对象，cv2.COLOR_BGR2GRAY则表面处理的类型为灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 形态学处理
    prc = Process(gray)
    # 得到顶点集
    regions = GetRegion(prc)
    # 打印顶点集的数量，即识别出来的绿框数量
    print('[INFO]:Detect %d license plates' % len(regions))
    for box in regions:
        # 绘制轮廓，根据提供的边界点绘制任何形状
        # img是需要做处理的图像，[box]是轮廓点，0是绘制的轮廓轮廓索引，(0,255,0)说明绘制的线是绿色，2是绘制的宽度
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv2.imshow('Result', img)
    # 保存结果文件名
    cv2.imwrite('result2.jpg', img)
    # 等待键盘按键响应
    cv2.waitKey(0)
    # 销毁窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 输入的参数为图片的路径
    img_path = input()
    img = cv2.imread(img_path)
    detect(img)