
import pandas as pd
import numpy as np
from osgeo import gdal
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有

import hyperpara as para ##引用参数
image_size = para.image_size
##transfrom labels

global cnt 
cnt = 0

def data_streamlined(): 

    s = pd.read_csv("./data/sample_train.txt")
    s = s.values

    print(s[:3, :])  # 样例输出
    print(s.shape)  # 输出形状

    # print(s[s[:, 3] != 3])  # 尺寸全等于3

    res = s[:, [2, 5, 6]]
    # print(res[:3, :])  # 样例输出    

    res[:, -1] = [round(-i) for i in res[:, -1]]
    res[:, -2] = [round(i) for i in res[:, -2]]
  
    res = pd.DataFrame(res, index=None, columns=None)
    res.to_csv("./data/train.txt", header=None, index=None)

    return res

def get_roat_tif_data(ss):

    res1 = pd.read_csv(ss,header=None)  
    tif_data = []
    for m, row in tqdm(list(res1.iterrows())):
        img = get_cell(row[1], row[2])
        tif_data.append([img, labels_key[row[0]]])

    data = np.array(tif_data)
    
    return data


def get_roat_cell(pos_x, pos_y):
    global cnt
    try:
        output = []
        for i in bands:
            band = dataset.GetRasterBand(i)
            t = band.ReadAsArray(int(pos_x - image_size / 2),
                                 int(pos_y - image_size / 2), image_size, image_size)
            # print(type(t))

            t =t.tolist()
            if cnt%3  == 0 :
                t[:] = map(list,zip(*t[::-1]))  ##矩阵旋转 90度

            if cnt%3  == 1 : 
                t[:] = t[::-1]

            if cnt%3  == 2 : 
                t[:] = t[:,::-1]

            if cnn ==10000:
                cnn =0


            output.append(np.array(t))
        img = np.array(output)
        
    except BaseException:
        return None
    return img

def get_tif_data(ss):

    res1 = pd.read_csv(ss,header=None)  
    tif_data = []
    for m, row in tqdm(list(res1.iterrows())):
        img = get_cell(row[1], row[2])
        tif_data.append([img, labels_key[row[0]]])
    data = np.array(tif_data)
    return data

def get_cell(pos_x, pos_y):
    try:
        output = []
        for i in bands:
            band = dataset.GetRasterBand(i)
            t = band.ReadAsArray(int(pos_x - image_size / 2),
                                 int(pos_y - image_size / 2), image_size, image_size)

            output.append(t)
        img = np.array(output)
        
    except BaseException:
        return None
    return img

dataset = gdal.Open(
    r"./data/GF6_WFV_E127.9_N46.8_20180823_L1A1119838015.tif")
bands = [i + 1 for i in range(8)]
print(bands)
print(dataset.RasterXSize)
print(dataset.RasterYSize)

labels_key = {'玉米': 0, '大豆': 1, '水稻': 2,'其他': 3}

if __name__ == '__main__':
	
    ss = "./data/train_sample_3col.txt"
    data1 = get_tif_data(ss)
    data2 = get_roat_tif_data(ss)  ##将图形旋转90度
    data = np.concatenate((data1,data2),axis=0)
    np.save("./data/train_sample.npy", data)

    # ss = "./data/sampling-data.txt"
    # data1 = get_tif_data(ss)
    # np.save("./data/sampling_data.npy", data1)


