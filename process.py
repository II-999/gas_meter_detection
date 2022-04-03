import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

digits = []  # 存放模板字


def load_digits():
    # 加载数字模板
    path = 'D:\\number'
    filename = os.listdir(path)
    i = 0
    for file in filename:
        i = i + 1
        img = cv2.imread(r'D:\\number\\' + file, cv2.IMREAD_GRAYSCALE)
        img_temp = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow('caijian'+str(i), img_temp)

        cnt = cv2.findContours(img_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # draw_img = cv2.drawContours(cv2.cvtColor(img_temp, cv2.COLOR_GRAY2BGR), cnt[0], -1, (0, 0, 255), 1)
        # cv2.imshow('caijian2', draw_img)
        # cv2.waitKey(0)
        x, y, w, h = cv2.boundingRect(cnt[0])
        digit_roi = cv2.resize(img_temp[y:y + h, x:x + w], (57, 88))
        # digit_roi = cv2.resize(img_temp, (57, 88))
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(digit_roi, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('caijian1'+str(i),closing)
        # cv2.waitKey(0)
        # 将数字模板存到列表中
        digits.append(closing)
        print("ok")
    return digits


load_digits()


# 模板匹配函数-返回匹配值
def demo(img):
    img_temp = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imshow('caijian', img_temp)
    cv2.waitKey(0)
    cnt = cv2.findContours(img_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    x, y, w, h = cv2.boundingRect(cnt[0])
    digit_roi = cv2.resize(img_temp[y:y + h, x:x + w], (57, 88))
    # digit_roi = cv2.resize(img_temp, (57, 88))
    digit_out = []
    source = []
    print("digit_roi", digit_roi)
    # print("digitROI", digitROI)
    for digitROI in digits:
        # print("digitROI:", digitROI)
        res = cv2.matchTemplate(
            digit_roi, digitROI, cv2.TM_CCOEFF_NORMED)
        max_val = cv2.minMaxLoc(res)[1]
        # print("res:", res)
        source.append(max_val)
        # print(res)
    if source:
        digit_out.append(str(source.index(max(source))))
        print(str(source.index(max(source))))
        return str(source.index(max(source)))


def fengge(img, below, result):
    def create_mask(image):  # 本身为去高光函数 255 255相当于未处理
        _, mask = cv2.threshold(image, 255, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
        return mask

    # 修复图片
    def xiufu(src_, mask):
        # 缩放因子(fx,fy)
        # res_ = cv2.resize(src_,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
        # mask = cv2.resize(mask,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
        dst = cv2.inpaint(src_, mask, 10, cv2.INPAINT_TELEA)
        return dst

    # 1、读取图像，并把图像转换为灰度图像并显示
    # img = cv2.imread(r"C:\Users\lenovo\Desktop\111.png")  # 读取图片
    mask = create_mask(img)
    imgg = xiufu(img, mask)
    img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)  # 转换了灰度化

    cv2.imshow('gray', img_gray)  # 显示图片
    cv2.waitKey(0)

    # 2、将灰度图像二值化，设定阈值是100
    img_thre = img_gray
    cv2.threshold(img_gray, below, 255, cv2.THRESH_BINARY_INV, img_thre)
    cv2.imshow('threshold', img_thre)
    cv2.waitKey(0)

    # 3、保存黑白图片
    cv2.imwrite('thre_res.png', img_thre)
    # cv2.imshow('thre_res', img_thre)

    # 4、分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)

    arg = False  # False表示白底黑字；True表示黑底白字

    # if black_max > white_max:
    #     arg = True

    # 分割图像
    def find_end(start_):
        end_ = start_ + 1
        for m in range(start_ + 1, width - 1):
            if (black[m] if arg else white[m]) > (
                    0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05
                end_ = m
                break
        return end_

    n = 1
    start = 1
    end = 2
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数请多调整，对应上面的0.95
            start = n
            end = find_end(start)
            n = end
            i = 0
            if end - start > 5:
                i = i + 1
                cj = img_thre[1:height, start:end]
                # cv2.imshow('caijian' + str(end) + '.jpg', cj)
                # cv2.imshow('caijian' + str(end), cj)
                # cv2.waitKey(0)
                result = result + str(demo(cj))
                # cv2.waitKey(0)
    return result


result = ""
img1 = cv2.imread("D:/27.jpg")
h, w, d = img1.shape
img2 = img1[:h - 2, :]
cv2.imshow("img2", img2)
result = fengge(img2, 110, result)
print(result)

# 1:109
