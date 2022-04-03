import os
import sys
import codecs
import shutil


def get_file_path(dir_name):
    listPath = []
    # root当前目录路径
    # dirs当前路径下所有子目录
    # files当前路径下所有非目录子文件
    for root, dirs, files in os.walk(dir_name):
        # 遍历文件
        for file in files:
            path = os.path.join(root, file)
            # print(path)
            # path为文件路径，封装在listPath中返回
            listPath.append(path)
    return listPath


def RemoveDir(filepath):
    # 如果文件夹不存在就创建，如果文件存在就清空！
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


def get_file_name(dir_name):
    listFileName = []
    for root, dirs, files in os.walk(dir_name):
        # 遍历文件
        for file in files:
            listFileName.append(file)
    return listFileName


def mkdir(path):
    if not os.path.exists(path):  # 判断是否存在文件夹,如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时,如果路径不存在会创建这个路径


def rename(path):
    old_names = os.listdir(path)
    for old_name in old_names:
        if old_name != sys.argv[0]:  # 代码本身文件路径，防止脚本文件放在路径文件下时，被一起重命名
            new_name = old_name.replace('resultframe', 'result')
            os.rename(os.path.join(path, old_name), os.path.join(path, new_name))
            print(old_name, "has been renamed successfully! New name is ", new_name)
