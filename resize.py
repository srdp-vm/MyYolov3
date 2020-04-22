# coding=utf-8
import os
import cv2
import glob

def resize_image(image, size=(500, 500)):
    """不改变图像长宽比，用padding填充，缩放image到size尺寸"""
    ih = image.shape[0]
    iw = image.shape[1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw, nh))
    image = cv2.copyMakeBorder(image, (h-nh)//2, h-nh-(h-nh)//2, (w-nw) //
                               2, w-nw-(w-nw)//2, cv2.BORDER_CONSTANT, value=[128, 128, 128])
    return image


if __name__ == "__main__":

    inputdir = "E:/Pisces/Pictures/"  # 输入文件夹 注意末尾有/
    outputdir = "F:/Photo_resize/"  # 输出文件夹 注意末尾有/
    if not os.path.exists(inputdir):
        print("输入路径不正确")
        exit(0)
    if not os.path.exists(outputdir): #输出文件夹是否存在
        os.mkdir(outputdir)
    for jpgfile in glob.glob(inputdir + "*.jpg"):
        print(jpgfile)
        origin = cv2.imread(jpgfile)
        resize = resize_image(origin)
        cv2.imwrite(os.path.join(outputdir, os.path.basename(jpgfile)), resize)
