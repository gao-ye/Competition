#-*- coding:utf-8 -*-

## 从原始样本中进行第一次采样
import pandas as pd
import numpy as np

radius = 15
sample_size = 5
labels_key = {'dadou': '大豆', 'shuidao': '水稻','yumi':'玉米'}

def data_sampling_1(name):
    data_sampling = []
    rectangle = pd.read_csv(name+'.txt',header=None)
    rectangle = rectangle.values
    rectangle = rectangle.astype(int)
    print(rectangle[:3,:])

    cnt = 0

    [rows, cols] = rectangle.shape
    for i in range(0, rows,2):
        if(i<5):
            sample_size = 8
        else:
            sample_size = 30

        # print(i)
        # cnt = 0

        # print (i)
        begin = rectangle[i+0, :]
        end = rectangle[i+1,:]

        for m in range(begin[0], end[0],sample_size):
            for n in range(begin[1], end[1],sample_size):
                # print("{} {}".format(m,n))
                cnt = cnt+1
                data_sampling.append(['其他' , m, n])

    print("cnt is{}".format( cnt))
    data_sampling = np.array(data_sampling)

    data_sampling = pd.DataFrame(data_sampling, index=None, columns=None)
    data_sampling.to_csv('./data/final-other-data.txt', header=None, index=None)
    print("ok")


def data_sampling(name):

    data = pd.read_csv(name)
    data = data.values
    print(data.shape)
    print(data[:3])

    data = data[:, [2, 5, 6]]
    # print(res[:3, :])  # 样例输出    
    data[:, -1] = [round(-i) for i in data[:, -1]]
    data[:, -2] = [round(i) for i in data[:, -2]]

    print(data[:3])
    print(type(data))

    temp_data = pd.DataFrame(data, index=None, columns=None)
    temp_data.to_csv('./data/train_sample_3col.txt', header=None, index=None)

    [rows,cols] = data.shape
    rectangle = []
    for i in range(rows):
        [n, x, y] = data[i,:]
        rectangle.append([n, int(x-radius), int(y-radius)])
        rectangle.append([n, int(x+radius), int(y+radius)])

    rectangle = np.array(rectangle)
    print(rectangle.shape)

    [rows, cols] = rectangle.shape
    cnt = 0 
    sampling =[]
    for i in range(0, rows,2):
        # print (i)
        [kind, x1, y1] = rectangle[i+0, :]
        [kind, x2, y2] = rectangle[i+1, :]

        for m in range(int(x1), int(x2),sample_size):
            for n in range(int(y1), int(y2),sample_size):

                cnt = cnt+1
                sampling.append([kind, m, n])

    print("cnt is{}".format( cnt))
    sampling = np.array(sampling)

    temp_data = pd.DataFrame(sampling, index=None, columns=None)
    temp_data.to_csv('./data/sampling-data.txt', header=None, index=None)


if __name__ == '__main__':

    ss = 'data/background'  #sample_size = 40  other  sample_size =50
    data_sampling_1(ss) ##从矩形区域采样

    # ss = './data/train_sample.txt'
    ss = './data/训练样本点.txt'
    data_sampling(ss) ##从矩形区域采样


