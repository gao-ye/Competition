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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import hyperpara as para ##引用参数
image_size = para.image_size  #the size of image to input model-net
batch_size = para.batch_size
model_name = para.model_name
x1         = para.x1
x2         = para.x2
x3         = para.x3
x4         = para.x4
x5         = para.x5
x6         = para.x6
x7         = para.x7

dataset = gdal.Open(
    r"./data/GF6_WFV_E127.9_N46.8_20180823_L1A1119838015.tif")
bands = [i + 1 for i in range(8)]
labels_key = [20, 60, 40, 0]


np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有



def get_batch_cell(pos_x, pos_y, batch_size):

    try:
        image = np.zeros([batch_size, 8, image_size, image_size])
        for k in range(batch_size):
            output = []
            for i in bands:
                band = dataset.GetRasterBand(i)
                t = band.ReadAsArray(int(pos_x - image_size / 2),
                                    int(pos_y - image_size / 2 + k*image_size ) , image_size, image_size)
                # print("get-cell{}---{}".format(pos_x - image_size / 2,pos_y - image_size / 2+ k*image_size))
                output.append(t)
            img = np.array(output)
            image[k]=img

    except BaseException:
        # print("error")
        return None
    return image    ## [batch, 8, size, size]

def test(name):
    # todo: 使用多线程、向量运算加速
    model = torch.load(model_name)
    model = model.cuda()
    # model = load_model("data/model_save.h5")
    print("the x is {}".format(dataset.RasterXSize))
    print("the y is {}".format(dataset.RasterYSize))
    res = np.zeros(
        shape=(
            dataset.RasterYSize,
            dataset.RasterXSize),
        dtype=np.uint8)
    cnt =0
    t = image_size // 2 
    correct_size = (batch_size,8, image_size, image_size)##正确的img 的维度大小

    RasterYSize = dataset.RasterYSize
    RasterXSize = dataset.RasterXSize

    m = RasterYSize %(image_size *batch_size)
    step = RasterYSize //(image_size *batch_size)
    change_point = step * image_size *batch_size + t
    small_batch_size = (RasterYSize - change_point) // image_size -1  ##到边缘时调整步长

    print(change_point)
    print(small_batch_size)

    x0 = t

    for i in tqdm(range(x1, x2, image_size)): ##遍历整张测试图集 接下来可以把整张测试
    #     print(i)
    # print("OK")
        change_batch_size = batch_size

        for j in range(t, RasterYSize, image_size *batch_size ):   ##图集加载进去 节省计算时间
            print(i,j)

            if j == change_point:
                change_batch_size = small_batch_size

            img = get_batch_cell(i, j, change_batch_size)
            if img is None:
                print("error1")
                continue

            img = np.asarray(img, dtype =  np.float32)
            E = operator.eq(img.shape, (change_batch_size,8, image_size, image_size))  ##此处判断是否正确取出img,可能由于数据损坏导致 img
            if E == False:                              ##不是 [1,8,5,5]的维度，无法送入网络测试，
                print("error2")
                continue                                ##如果格式错误，直接跳入下一循环

            img=torch.from_numpy(img)

            result = model.forward((img.cuda()))   ## 预测
            result =result.cpu().data.numpy()  ##to numpy 方便判断

            for n in range (change_batch_size):
                if np.max(result[n]) < 0.5:
                    color = 0
                else:
                    color = labels_key[np.argmax(result[n])]
                    cnt =cnt +1
                res[j-t + image_size*n:j + t +image_size*n, i - t:i + t] = color

    np.save("./code/np-data/001-np.npy", res)
    print("write successful")

if __name__ == '__main__':
    test("test_result")
    # data_connect()




