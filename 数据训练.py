# -*- coding: utf-8 -*-
# @File:数据训练.py
# @Author: Zhang Ze
# @Date:   2024-07-24
# @Last Modified by:   Zhang Ze
import datetime
import os
import time

import cv2
import numpy as np


def get_image_label(path):
    image_paths = []  # 所有图片路径
    face_samples = []  # 人脸数据
    ids = []  # 身份编号
    labels_list = []  # 模型标签编号
    # 加载分类器
    face_cascade = cv2.CascadeClassifier(
        'D:/PyProjects/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    # 获取faces路径下的所有文件夹
    dirs = os.listdir(path)

    # os.listdir()返回指定路径下的文件和文件夹列表
    # 获取每一个文件夹下的图片路径
    for f in dirs:
        f_path = os.path.join(path, f)
        for i in os.listdir(f_path):
            id = int(f.split('_')[0])
            ids.append(id)
            image_paths.append(os.path.join(f_path, i))

    # 遍历图片
    for n, imagePath in enumerate(image_paths):
        index = ids[n]
        img = cv2.imread(imagePath)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 人脸检测
        face = face_cascade.detectMultiScale(gray_img)
        # 遍历人脸
        for (x, y, w, h) in face:
            # 添加人脸数据
            face_samples.append(gray_img[y:y + h, x:x + w])
            labels_list.append(index)

    # 返回人脸数据和模型标签编号
    return face_samples, labels_list


if __name__ == '__main__':
    # 模型路径
    path = './faces/'
    # 获取人脸数据和身份信息
    faces, ids = get_image_label(path)
    # 导入人脸识别模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # # 训练模型
    recognizer.train(faces, np.array(ids))
    trainer_date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # 保存模型
    recognizer.save('./face_yml/{0}.yml'.format(trainer_date))
