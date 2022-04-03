import assess
import script
import numpy as np
import cv2


def pre_treatment2(filepath, Whether_show=True, Filename='888.jpg', Blur=9):
    # 保存中间处理过程
    images = []
    titles = []

    # 读入图片
    try:
        # print(filepath)
        Original = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
    except IOError:
        print('文件路径错误！')
        return False
    images.append(Original)
    titles.append('Original')

    # 图片分辨率调整
    Resize = cv2.resize(Original, (100, 150), interpolation=cv2.INTER_AREA)
    Resize = Original
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

    # 使用黑帽运算突出比原轮廓暗的部分，让数字区成为一个整体
    kernel2 = np.ones((40, 40), np.uint8)  # kernel越大，越不黑的的地方也能分割
    BlackHat = cv2.morphologyEx(Gray, cv2.MORPH_BLACKHAT, kernel2)
    images.append(BlackHat)
    titles.append('BlackHat')

    # 二值化求阈值
    ret, Binary = cv2.threshold(BlackHat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    cv2.imwrite('./num_result/' + Filename, Close)

    # 显示图片
    mid_save_path = './num_mid/'
    if Whether_show:
        for i in range(len(titles) - 7):
            cv2.imshow(titles[i + 7], images[i + 7])
            cv2.waitKey(0)
    script.view_photos(images, titles, mid_save_path, Filename, Whether_show)
    return Close


if __name__ == '__main__':
    # 处理脚本
    # assess.RemoveDir('./num_result/')
    path = './number/'
    filename = '1.jpg'
    path += filename
    # 文件夹下所有文件路径与文件名
    file_paths = assess.get_file_path(path)
    file_names = assess.get_file_name(path)
    if len(file_paths) == 0:
        whether_show = False  # 需要展示细节工作
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
