# windows:
# https://digi.bib.uni-mannheim.de/tesseract/
# 配置环境变量如E:\Program Files (x86)\Tesseract-OCR
# tesseract -v 查看版本进行测试
# tesseract XXX.png 得到结果
# pip install pytesseract
# anaconda lib site-packges pytesseract pytesseract.py
# tesseract_cmd 修改为绝对路径即可

# ubuntu:
# pip install pytesseract
# sudo apt install tesseract-ocr
from PIL import Image
import pytesseract
import cv2
import os

preprocess = 'blur'  # thresh

image = cv2.imread('scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理
if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# 中值滤波
if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)

# 以系统进程号.png为名保存图片
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
