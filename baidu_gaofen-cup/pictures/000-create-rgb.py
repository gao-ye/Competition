
import pandas as pd
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import cv2 

np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有

import hyperpara as para ##引用参数
image_size = para.image_size
##transfrom labels

global cnt 
cnt = 0

def getcell(pos_x, pos_y):

    output = []


    h_size = 17000
    w_size = 50000

    data = np.zeros(
        shape = (h_size, w_size,3),
        dtype = np.int64
    )

    for i in range(3):

        m = dic[i]
        print(m)
        band = dataset.GetRasterBand(m)
        print(band.GetNoDataValue())
        t = band.ReadAsArray(int(1),
                                int(1), w_size, h_size)
        data[:, :, i] = t
    return data

dataset = gdal.Open(
    r"./data/GF6_WFV_E127.9_N46.8_20180823_L1A1119838015.tif")
# bands = [8, 5, 1]
bands = [8]
dic ={0:8, 1:5, 2:1}
print(bands)
print(dataset.RasterXSize)
print(dataset.RasterYSize)

labels_key = {'玉米': 0, '大豆': 1, '水稻': 2,'其他': 3}

if __name__ == '__main__':
	
    img = cv2.imread("rgb.jpg", 1)
    data = img[0:5, 0:3, :]
    print(data)
    print(data.shape)




    img = getcell(0,0)
    img = img *255/6553.0
    # print(type(img[1,1,1]))
    # print(img)
    # print(img.shape)

    cv2.imwrite("./data/rgb.jpg", img)


