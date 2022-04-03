import cv2
import numpy as np
import matplotlib.pyplot as plt
import assess


# 图片预览
def view_photos(images, titles, Save_path, Filename, show=False):
    n = len(titles)
    row = column = int(n ** 0.5)
    while row * column < n:
        column += 1
    plt.figure()
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(row):
        for j in range(column):
            if (i * column + j) < n:
                Original = images[i * column + j]
                title = titles[i * column + j]
                # 行，列，索引
                plt.subplot(row, column, i * column + j + 1)
                # CV读取的是BGR，转换成RGB
                if len(Original.shape) == 3:
                    Original = Original[:, :, [2, 1, 0]]
                    plt.imshow(Original)
                else:
                    plt.imshow(Original, cmap='gray')
                plt.title(title, fontsize=9)
                plt.xticks([])
                plt.yticks([])
    assess.mkdir(Save_path)
    plt.savefig(Save_path + Filename)
    if show:
        plt.show()


# 确定上分界线
def find_up(col_num, row_num, img_hsv):
    up = 0
    for y in range(col_num):
        black = 0
        for x in range(row_num):
            h = img_hsv.item(y, x, 0)
            s = img_hsv.item(y, x, 1)
            v = img_hsv.item(y, x, 2)
            # 判断黑色
            if h <= 180 and s <= 255 and v <= 46:
                black += 1
        if black * 2 >= row_num:
            up = y + 5
            return up
    return up


# 确定下分界线
def find_down(col_num, row_num, img_hsv):
    down = col_num
    for y in range(col_num):
        black = 0
        for x in range(row_num):
            h = img_hsv.item(y, x, 0)
            s = img_hsv.item(y, x, 1)
            v = img_hsv.item(y, x, 2)
            # 判断黑色
            if h <= 180 and s <= 255 and v <= 46:
                black += 1
        if black * 2 >= row_num:
            down = y - 10
    return down


# 确定左分界线
def find_left(col_num, row_num, img_hsv):
    left = 0
    for x in range(row_num):
        black = 0
        for y in range(col_num):
            h = img_hsv.item(y, x, 0)
            s = img_hsv.item(y, x, 1)
            v = img_hsv.item(y, x, 2)
            # 判断黑色
            if h <= 180 and s <= 255 and v <= 46:
                black += 1
        if black * 2 >= col_num:
            left = x + 5
            return left
    return left


# 确定右分界线
def find_right(col_num, row_num, img_hsv):
    right = row_num
    for x in range(row_num):
        red = 0
        for y in range(col_num):
            h = img_hsv.item(y, x, 0)
            s = img_hsv.item(y, x, 1)
            v = img_hsv.item(y, x, 2)
            # 判断红色
            if (0 <= h <= 10 or 156 <= h <= 180) and 43 <= s and 46 <= v:
                red += 1
        if red * 1.3 >= col_num:
            right = x - 9
    return right


# 确定黑红分界线
def find_b_r(col_num, row_num, img_hsv):
    b_r = row_num * 0.75
    for x in range(row_num):
        red = 0
        for y in range(col_num):
            h = img_hsv.item(y, x, 0)
            s = img_hsv.item(y, x, 1)
            v = img_hsv.item(y, x, 2)
            # 判断红色
            if (0 <= h <= 10 or 156 <= h <= 180) and 43 <= s and 46 <= v:
                # print(x, y)
                red += 1
        if red * 2 >= col_num:
            b_r = x
    return b_r


# gamma函数处理亮度
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


# 数字区边界修正
def find_boundary(img):
    # 图片增亮
    img_trans = gamma_trans(img, 2.3)
    # 图片转换
    img_hsv = cv2.cvtColor(img_trans, cv2.COLOR_BGR2HSV)
    col_num, row_num = img_hsv.shape[:2]
    # 寻找边界
    up_ = find_up(col_num, row_num, img_hsv)
    down_ = find_down(col_num, row_num, img_hsv)
    left_ = find_left(col_num, row_num, img_hsv)
    right_ = find_right(col_num, row_num, img_hsv)
    b_r_ = find_b_r(col_num, row_num, img_hsv)
    return up_, down_, left_, right_, b_r_


def __point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# 预处理
def pre_treatment1(filepath, Whether_show=True, Filename='888.jpg', Blur=9, Min_area=4000, Max_width=600):
    # 保存中间处理过程
    images = []
    titles = []
    # 读入图片
    try:
        Original = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
    except IOError:
        print('文件路径错误！')
        return False
    Original_height, Original_width = Original.shape[:2]
    images.append(Original)
    titles.append('Original')

    # 图片分辨率调整
    if Original_width > Max_width:
        resize_rate = Max_width / Original_width
        Resize1 = cv2.resize(Original, (Max_width, int(Original_height * resize_rate)), interpolation=cv2.INTER_AREA)
    else:
        Resize1 = Original
    images.append(Resize1)
    titles.append('Resize1')

    # 锐化
    kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    Sharpen = cv2.filter2D(Resize1, -1, kernel=kernel1)
    images.append(Sharpen)
    titles.append('Sharpen')

    # 高斯模糊
    if Blur > 0:
        GaussianBlur = cv2.GaussianBlur(Sharpen, (Blur, Blur), 0)
    else:
        GaussianBlur = Sharpen
    images.append(GaussianBlur)
    titles.append('GaussianBlur')

    # 保存原图
    Source = Resize1
    images.append(Source)
    titles.append('Source')

    # 转灰度图
    Gray = cv2.cvtColor(GaussianBlur, cv2.COLOR_BGR2GRAY)
    images.append(Gray)
    titles.append('Gray')

    # 使用黑帽运算突出比原轮廓暗的部分，让数字区成为一个整体
    kernel2 = np.ones((30, 30), np.uint8)
    BlackHat = cv2.morphologyEx(Gray, cv2.MORPH_BLACKHAT, kernel2)
    images.append(BlackHat)
    titles.append('BlackHat')

    # 二值化求阈值
    ret, Binary = cv2.threshold(BlackHat, 0, 255, cv2.THRESH_OTSU)
    images.append(Binary)
    titles.append('Binary')

    # 闭运算
    kernel3 = np.ones((2, 15), np.uint8)
    Close = cv2.morphologyEx(Binary, cv2.MORPH_CLOSE, kernel3)
    images.append(Close)
    titles.append('Close')

    # 开运算
    kernel4 = np.ones((4, 30), np.uint8)
    Open = cv2.morphologyEx(Close, cv2.MORPH_OPEN, kernel4)
    images.append(Open)
    titles.append('Open')

    # 计算边缘
    Edge = cv2.Canny(Open, 100, 200)
    images.append(Edge)
    titles.append('Edge')

    try:
        image, contours, hierarchy = cv2.findContours(Edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        # ValueError: not enough values to unpack (expected 3, got 2)
        # cv2.findContours方法在高版本OpenCV中只返回两个参数
        contours, hierarchy = cv2.findContours(Edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_area]

    Gray = cv2.cvtColor(Gray, cv2.COLOR_GRAY2BGR)
    Selected = Gray.copy()
    # 逐个排除不是数字的矩形区域
    number_contours = []
    for cnt in contours:
        # 框选，生成最小外接矩形，返回值(中心(x,y),(宽,高),旋转角度)
        rect = cv2.minAreaRect(cnt)
        # print('宽高:', rect[1])
        width, height = rect[1]

        # 选择宽大于高的区域
        if width < height:
            width, height = height, width
        wh_ratio = width / height
        # print('宽高比：', wh_ratio)

        # 6到8是数字区的长宽比，其余的矩形排除
        # if 6 < wh_ratio < 8:
        if 1:
            number_contours.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            Selected = cv2.drawContours(Gray, [box], 0, (0, 0, 255), 2)

    # cv2.imshow('Rects', Selected)
    # cv2.waitKey(0)
    if len(number_contours):
        images.append(Selected)
        titles.append('Selected')

    # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
    number_images = []
    for rect in number_contours:
        if -1 < rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免数字边缘被排除
        box = cv2.boxPoints(rect)
        high_point = right_point = [0, 0]
        left_point = low_point = [Original_width, Original_height]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if high_point[1] < point[1]:
                high_point = point
            if right_point[0] < point[0]:
                right_point = point

        # 正角度
        if left_point[1] <= right_point[1]:
            new_right_point = [right_point[0], high_point[1]]
            pts2 = np.float32([left_point, high_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, high_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)  # 获得变换矩阵M
            dst = cv2.warpAffine(Gray, M, (Original_width, Original_height))  # 仿射变换
            __point_limit(new_right_point)
            __point_limit(high_point)
            __point_limit(left_point)
            num_img = dst[int(left_point[1]):int(high_point[1]), int(left_point[0]):int(new_right_point[0])]
            number_images.append(num_img)

        # 负角度
        elif left_point[1] > right_point[1]:
            new_left_point = [left_point[0], high_point[1]]
            pts2 = np.float32([new_left_point, high_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, high_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)  # 获得变换矩阵M
            dst = cv2.warpAffine(Source, M, (Original_width, Original_height))  # 仿射变换
            __point_limit(right_point)
            __point_limit(high_point)
            __point_limit(new_left_point)
            num_img = dst[int(right_point[1]):int(high_point[1]), int(new_left_point[0]):int(right_point[0])]
            number_images.append(num_img)

    final_img = cv2.resize(number_images[0], (300, 50), interpolation=cv2.INTER_CUBIC)

    # 利用颜色定位，排除不是数字区域的矩形
    for Number_index, Number_img in enumerate(number_images):
        red = black = gray = white = 0
        # noinspection PyBroadException
        try:
            # 有转换失败的可能，原因来自于上面矫正矩形出错
            Number_img_hsv = cv2.cvtColor(Number_img, cv2.COLOR_BGR2HSV)
        except Exception:
            Number_img_hsv = None

        if Number_img_hsv is None:
            continue
        col_num, row_num = Number_img_hsv.shape[:2]
        count = row_num * col_num

        # 确定数字区颜色
        for x in range(row_num):
            for y in range(col_num):
                h = Number_img_hsv.item(y, x, 0)
                s = Number_img_hsv.item(y, x, 1)
                v = Number_img_hsv.item(y, x, 2)
                # 颜色计数
                if (0 <= h <= 10 or 156 <= h <= 180) and 43 <= s and 46 <= v:
                    red += 1
                if h <= 180 and s <= 255 and v <= 46:
                    black += 1
                elif h <= 180 and s <= 43 and 46 < v <= 220:
                    gray += 1
                elif h <= 180 and s <= 30 and 221 <= v:
                    white += 1

        # 确认颜色为黑+红才输出
        if (black + gray + white) * 3 >= count and red * 5 >= count:
            # print('红：{:<3}黑：{:<3}灰：{:<3}白：{:<3}总数：{:<3}'.format(red, black, gray, white, count))
            images.append(number_images[Number_index])
            titles.append('Color_img' + str(Number_index))

            # 根据数字区颜色再定位，边界修正
            # 图片分辨率调整
            Resize2 = cv2.resize(Number_img, (300, 50), interpolation=cv2.INTER_CUBIC)
            up, down, left, right, b_r = find_boundary(Resize2)
            if up >= down:
                print(1)
                down = col_num
            if left >= right:
                print(2)
                right = row_num
            # print(up, down, left, right)

            # 获得固定区域
            final_img = Resize2[up:down, left:right]
            # 保存识别窗口
            final_path = './final_result/'
            file_name = final_path + str(Number_index) + '__' + Filename
            cv2.imwrite(file_name, final_img)

            images.append(final_img)
            titles.append('Final_img' + str(Number_index))
            # black = Number_img[:, :int(b_r)]
            # red = Number_img[:, int(b_r):]

    # 显示图片
    mid_save_path = './mid_result/'
    ls = 0
    if Whether_show:
        for i in range(len(titles) - ls):
            cv2.imshow(titles[i + ls], images[i + ls])
            cv2.waitKey(0)
    view_photos(images, titles, mid_save_path, Filename, Whether_show)
    # 返回是否找到数字区域
    if len(titles) > 12:
        return final_img
    else:
        return False


def pre_treatment2(filepath, Whether_show=True, Filename='888.jpg', Blur=9):
    # 保存中间处理过程
    images = []
    titles = []

    # 读入图片
    try:
        Original = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
    except IOError:
        print('文件路径错误！')
        return False
    images.append(Original)
    titles.append('Original')

    # 图片分辨率调整
    Resize = cv2.resize(Original, (300, 50), interpolation=cv2.INTER_AREA)
    images.append(Resize)
    titles.append('Resize')

    # 锐化
    kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    Sharpen = cv2.filter2D(Resize, -1, kernel=kernel1)
    images.append(Sharpen)
    titles.append('Sharpen')

    # 高斯模糊
    if Blur > 0:
        GaussianBlur = cv2.GaussianBlur(Sharpen, (Blur, Blur), 0)
    else:
        GaussianBlur = Sharpen
    images.append(GaussianBlur)
    titles.append('GaussianBlur')

    # 保存原图
    Source = Resize
    images.append(Source)
    titles.append('Source')

    # 转灰度图
    Gray = cv2.cvtColor(GaussianBlur, cv2.COLOR_BGR2GRAY)
    images.append(Gray)
    titles.append('Gray')

    # 顶帽(top-hat)，突出比原轮廓亮的部分，突出数字
    kernel2 = np.ones((20, 20), np.uint8)  # 25, 5 横明纵暗，有字符分割效果
    TopHat = cv2.morphologyEx(Gray, cv2.MORPH_TOPHAT, kernel2)
    images.append(TopHat)
    titles.append('TopHat')

    # 二值化求阈值
    ret, Binary = cv2.threshold(TopHat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images.append(Binary)
    titles.append('Binary')

    # 开运算
    kernel3 = np.ones((3, 3), np.uint8)
    Open = cv2.morphologyEx(Binary, cv2.MORPH_OPEN, kernel3)
    images.append(Open)
    titles.append('Open')

    # 闭运算
    kernel4 = np.ones((2, 2), np.uint8)
    Close = cv2.morphologyEx(Open, cv2.MORPH_CLOSE, kernel4)
    images.append(Close)
    titles.append('Close')
    cv2.imwrite('./binary/' + Filename, Close)

    # 显示图片
    x = 0
    mid_save_path = './test/'
    if Whether_show:
        for i in range(len(titles) - x):
            cv2.imshow(titles[i + x], images[i + x])
            cv2.waitKey(0)
    view_photos(images, titles, mid_save_path, Filename, Whether_show)
    return Close


if __name__ == '__main__':
    assess.RemoveDir('./final_result/')
    assess.RemoveDir('./mid_result/')

    path = './images/'
    filename = '999.jpg'
    path += filename
    # 文件夹下所有文件路径与文件名
    file_paths = assess.get_file_path(path)
    file_names = assess.get_file_name(path)
    if len(file_paths) == 0:
        whether_show = True  # 需要展示细节工作
        flag = pre_treatment1(path, whether_show, filename)
        if flag is False:
            print("失败")
        else:
            print(filename, True)
    else:
        whether_show = False  # 不需要展示细节工作
        for index, file_path in enumerate(file_paths):
            flag = pre_treatment1(file_path, whether_show, file_names[index])
            if flag is False:
                print("失败")
            else:
                print(file_names[index], True)

    # 处理脚本
    assess.RemoveDir('./test/')
    assess.RemoveDir('./binary/')
    path = './final_result/'
    filename = '0__22.jpg'
    # path += filename
    # 文件夹下所有文件路径与文件名
    file_paths = assess.get_file_path(path)
    file_names = assess.get_file_name(path)
    if len(file_paths) == 0:
        whether_show = True  # 需要展示细节工作
        flag = pre_treatment2(path, whether_show, filename)
        if flag is False:
            print("失败")
        else:
            print(filename, True)
    else:
        whether_show = False  # 不需要展示细节工作
        for index, file_path in enumerate(file_paths):
            flag = pre_treatment2(file_path, whether_show, file_names[index])
            if flag is False:
                print("失败")
            else:
                print(file_names[index], True)
