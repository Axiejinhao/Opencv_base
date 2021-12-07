# 导入工具包
import utils_paths
import numpy as np
import cv2

# 标签文件处理
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# Caffe所需配置文件
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
                               "bvlc_googlenet.caffemodel")

# 图像路径
imagePaths = sorted(list(utils_paths.list_images("images/")))

# 图像数据预处理
image = cv2.imread(imagePaths[0])
resized = cv2.resize(image, (224, 224))

# image scalefactor size mean swapRB
# RGB的均值(104, 117, 123)

"""
cv2.dnn.blobFromImage(image,scalefactor,size,mean,swapRB,crop,ddepth)
对图像进行预处理,包括减均值,比例缩放,裁剪,交换通道等
返回一个4通道的blob(blob可以简单理解为一个N维的数组,用于神经网络的输入)
    参数:
    image:输入图像（1、3或者4通道）
    可选参数:
    scalefactor:图像各通道数值的缩放比例
    size:输出图像的空间尺寸,如size=(200,300)表示高h=300,宽w=200
    mean:用于各通道减去的值，以降低光照的影响(e.g. image为bgr3通道的图像，mean=[104.0, 177.0, 123.0],表示b通道的值-104，g-177,r-123)
    swapRB:交换RB通道，默认为False.(cv2.imread读取的是彩图是bgr通道)
    crop:图像裁剪,默认为False.当值为True时，先按比例缩放，然后从中心裁剪成size尺寸
    ddepth:输出的图像深度，可选CV_32F或者CV_8U.
"""
blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

# 得到预测结果
net.setInput(blob)
preds = net.forward()

# 排序,取分类可能性最大的
idx = np.argsort(preds[0])[::-1][0]
text = "Label: {}, {:.2f}%".format(classes[idx],
                                   preds[0][idx] * 100)
cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

# 显示
cv2.imshow("Image", image)
cv2.waitKey(0)

# Batch数据制作
images = []

# 方法一样,数据是一个batch
for p in imagePaths[1:]:
    image = cv2.imread(p)
    image = cv2.resize(image, (224, 224))
    images.append(image)

# blobFromImages函数,注意有s
blob = cv2.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
print("Second Blob: {}".format(blob.shape))

# 获取预测结果
net.setInput(blob)
preds = net.forward()
for (i, p) in enumerate(imagePaths[1:]):
    image = cv2.imread(p)
    idx = np.argsort(preds[i])[::-1][0]
    text = "Label: {}, {:.2f}%".format(classes[idx],
                                       preds[i][idx] * 100)
    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
