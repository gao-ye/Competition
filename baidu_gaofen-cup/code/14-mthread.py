# -*- coding: utf-8 -*-
#pytorch.__version__ = 0.4.0

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from osgeo import gdal
from tqdm import tqdm
import cv2
import operator

from resnet import ResNet

model =ResNet()
model = model.cuda()

import hyperpara as para ##引用参数
image_size = para.image_size  #the size of image to input model-net
batch_size = para.batch_size
x1         = para.x1
x2         = para.x2
x3         = para.x3
RasterXSize =para.RasterXSize
RasterYSize =para.RasterYSize

t = image_size // 2

np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有

# r g b 

# yumi 0 20 [255, 0, 0] b
# shuidao 1 60 [0, 0, 255]   r
# dadou   2   40  [0, 255, 0] g 

rgb_key = {0:[0,0,0], 20: [255, 0, 0], 40: [0, 255, 0], 60:[0, 0, 255]}

def show_data():
    # todo: 使用多线程、向量运算加速



    for i in tqdm(range(t, RasterXSize, image_size)): ##遍历整张测试图集 接下来可以把整张测试
        print(i)
    print("the divided data has been written")

def  data_connect(name):
    data = np.zeros(
    shape=(
        RasterYSize,
        RasterXSize),
    dtype=np.uint8)

    data0 = np.load("./code/np-data/000-np.npy")
    data1 = np.load("./code/np-data/001-np.npy")
    data2 = np.load("./code/np-data/002-np.npy")
    data3 = np.load("./code/np-data/003-np.npy")
 
 ##  x1 - image_size +t
    data[:,:x1-image_size+t] = data0[:,:x1-image_size +t ]
    data[:,x1-image_size+t:x2-image_size +t ] =data1[:,x1-image_size+t:x2-image_size +t ]
    data[:,x2-image_size+t:x3-image_size +t ] =data2[:,x2-image_size+t:x3-image_size +t ]
    data[:,x3-image_size+t:] =data3[:,x3-image_size+t:]

    # np.save('np-data/final-np.npy',data)

    cv2.imwrite("./code/result/{}.tif".format(name), data)
    cv2.imwrite("./code/result/{}.jpg".format(name), data)

    #    
    # cv2.imwrite("result/000.jpg", data0)
    # cv2.imwrite("result/001.jpg", data1)
    # cv2.imwrite("result/002.jpg", data2)
    # cv2.imwrite("result/003.jpg", data3)

    print("write successful")

    return 



if __name__ == '__main__':
    import os
    os.system("mkidr -p ./code/result")
    # show_data()
    data_connect("test_result")
    # change_rgb()




