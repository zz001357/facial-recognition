# -*- coding: utf-8 -*-
# @File:尺寸转换.py
# @Author: Zhang Ze
# @Date:   2024-07-22
# @Last Modified by:   Zhang Ze


# 1.导入cv模块
import cv2

# 2.导入图片
img = cv2.imread('./img/_DSF1297.jpg')

# 2.1修改尺寸
resize = cv2.resize(src=img, dsize=(200, 200))

# 3.显示原始图片
cv2.imshow("img", img)

# 3.1显示修改图片
cv2.imshow("resize", resize)

# 4.等待.毫秒为单位，0代表无限等待，直到按键事件发生
cv2.waitKey(0)

# 5.释放内存,关闭窗口
cv2.destroyAllWindows()
