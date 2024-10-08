@[toc](目录)

# 目录

## 人脸检测识别的流程

### [人脸录入](./人脸录入.py)

### [训练数据](./数据训练.py)

### [人脸识别](./人脸识别.py)

## 提高OpenCV人脸识别精度方法

### 1. 选择合适的人脸检测器

-  不同人脸检测器区别

> **Haar特征**、**Haar特征的改进版本（Haar_alt和Haar_alt2）**、**LBP特征**等。
> **Haar_alt2** 在简单背景图像中检测效果较好。   
> **LBP** 在复杂背景图像中的检测效果较好

### 2.使用detectMultiScale函数

> 使用detectMultiScale函数：这个函数可以进一步优化人脸检测。通过调整函数参数，如scaleFactor（比例系数），minNeighbors（最小邻居数），flags（标志位）等，可以提高检测的准确性。例如，适当增加minSize的值可以减少误检的小方框。  

### 3.数据预处理

> 在人脸识别之前，对输入的人脸图像进行预处理，如灰度化、二值化、去噪等，可以减少图像中的噪声和干扰，提高人脸识别的精度。
> 在需要保留更多亮度信息的情况下 **灰度化** 是更合适的选择。而在需要简化图像以便于特征提取或形态学分析时 **二值化** 则更为适用
>
> - **二值化** 的目标是将图像中的信息从背景中分离出来，以便更容易进行特征提取、分割和分析。图像二值化的主要思想是根据像素的灰度值将图像分为两个不同的区域：前景和背景。前景通常包含我们感兴趣的目标对象，而背景则是目标对象的周围环境。通过将前景和背景分离成明显不同的像素值，可以更容易地进行目标检测、分割、轮廓提取等图像处理任务。  
>   字符识别：在光学字符识别（OCR）和文本分析中，图像二值化用于将文本从图像中提取出来，以便进行字符识别。
>   目标检测：在计算机视觉中，二值化有助于分离目标对象（如人脸、车辆等）和背景 ，从而更容易进行目标检测和跟踪。
>   图像分割：图像二值化是图像分割的预处理步骤，用于将图像分为多个区域，以便分析各个区域的特征。
>   文档处理：在文档图像处理中，二值化用于将文本、图形和背景分离，以便进行文档分析和处理。
> - **灰度化** 只包含亮度信息，而没有颜色信息

### 4.数据训练要完整，尽可能在数据上做规整

## 置信度confidence 过大

### 1. 没有检测到人脸

图片畸形是一个问题！

调整scaleFactor比例系数，与minNeighbors 系数。两个参数值越大限制条件越严格
minSize 与 maxSize 影响人脸检测大小从而间接影响识别

### 2.检测范围过大，检测到别的地方，产生误检

### 3.可能训练的时候没有把数据训练完整

### 4.数据量过小或者不统一！（这个很关键）

