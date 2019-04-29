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
from resnet import ResNet

class_num = 4
learning_rate=0.0001



import hyperpara as para ##引用参数
image_size = para.image_size  #the size of image to input model-net
batch_size = 500

labels_key = [20, 40, 60, 0]

tra_size = 2372 * 2 
break_batch_tra = int(tra_size/batch_size)


np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有

dataset = gdal.Open(
    r"data/GF6_WFV_E127.9_N46.8_20180823_L1A1119838015.tif")
bands = [i + 1 for i in range(8)]

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


def get_roat_cell(pos_x, pos_y):
    global cnt
    try:
        output = []
        for i in bands:
            band = dataset.GetRasterBand(i)
            t = band.ReadAsArray(int(pos_x - image_size / 2),
                                 int(pos_y - image_size / 2), image_size, image_size)
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


def get_roat_tif_data(ss):

    res1 = pd.read_csv(ss,header=None)  
    tif_data = []
    for m, row in tqdm(list(res1.iterrows())):
        img = get_cell(row[1], row[2])
        tif_data.append([img, labels_key[row[0]]])

    data = np.array(tif_data)
    
    return data

def get_tif_data_５(ss):
    res1 = pd.read_csv(ss,header=None)  
    tif_data = []
    for m, row in tqdm(list(res1.iterrows())):
        img = get_cell(row[1], row[2])
        tif_data.append([img, labels_key[row[0]], row[1], row[2]])
    data = np.array(tif_data)
    return data


def load_data():
    data = np.load("./data/train_sample.npy")
    np.random.shuffle(data)
    y= data[:,1]
    return np.array([i for i in data[:,0]]),y

def train():

    x,y = load_data()
    model =ResNet()
    model = model.cuda()
    # Loss and Optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    
    x = np.asarray(x, dtype =  np.int32)  
    y = np.asarray(y, dtype =  np.int32)  
    x_torch=torch.from_numpy(x)
    y_torch=torch.from_numpy(y)

    # 先转换成 torch 能识别的 Dataset
    torch_dataset = Data.TensorDataset(x_torch, y_torch)
    
    # 把 dataset 放入 DataLoader
    train_loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=batch_size,      # mini batch size
        shuffle=False,               # 要不要打乱数据 (打乱比较好)
        num_workers=1,              # 多线程来读数据
    )

    #traing the  model
    for epoch in range(15):
        model.train()
        print("epoch {}".format(epoch))
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            #print(images.shape)
            labels = Variable(labels)
            # print(images.shape)
            # print(labels.shape)
            
		    # #forward + backward + optimizer
            optimizer.zero_grad()
            images=images.float()

            labels=labels.long()
            if i == break_batch_tra :        #进行到最后一个batch_size时, reshape 无法
                break                   # 使用，直接跳过
           
            labels = labels.reshape(batch_size,1)      
            labels = torch.zeros(batch_size, class_num).scatter_(1, labels, 1) ##one_hot 编码
            
            outputs = model(images.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()


        if  1 :
            model.eval()
            total = 0
            correct = 0
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images)
                labels = Variable(labels)
                images=images.float()
                labels=labels.long()
                if i == break_batch_tra :        #进行到最后一个batch_size时, reshape 无法
                    break                   # 使用，直接跳过
                labels = labels.reshape(batch_size)      

                outputs = model(images.cuda())
                pred_data, pred = torch.max(outputs.data,1)		##取出经过网络计算后的标签
                test = outputs.data

                pred = pred.cpu().numpy()
                labels = labels.numpy()
                correct += (pred == labels).sum()
                total += np.size(labels,0)
            accu = correct/total
            print("train correct is {}".format(correct))
            print("train total is {}".format(total))
            print("train accuracy is {}\n".format(accu))

        # torch.save(model, './model/'+str(epoch)+'test-model.pkl')
            
    torch.save(model, './model/test-model.pkl')
    print("model has been written")

    return 

def test():
    # todo: 使用多线程、向量运算加速
    model = torch.load('./model/test-model.pkl')
    model = model.cuda()

    cnt =0
    data = np.load("./data/sampling_data.npy")

    print (data.shape)
    print (data[0].shape)

    [rows, cols] = data.shape

    valid_label = []
    valid_tra   = []

    for i in tqdm(range(0, rows)):

        img   = data[i,0]
        label = data[i, 1]
        img = np.asarray([img], dtype =  np.float32)
        img=torch.from_numpy(img)

        result = model.forward((img.cuda()))   ## 预测
        result =result.cpu().data.numpy()  ##to numpy 方便判断
        result = result[0]

        ## 分类正确 且 置信度大于 0.9 则将label 坐标记录下来
        if label == np.argmax(result) and np.max(result) >=0.75:
            valid_label.append([label , data[i, 2], data[i, 3]])
            valid_tra.append([data[i,0], data[i,1]])
     
        
    valid_label = np.array(valid_label)
    # print(valid_label.shape)

    valid_tra = np.array(valid_tra)
    # print(valid_tra.shape)

    np.save("./data/valid_label.npy", valid_label)
    # np.save("data/valid_tra.npy", valid_tra)

def create_final_train_data():

    data = np.load("./data/valid_label.npy")
    lab_key = {0:'玉米', 1:'大豆', 2:'水稻'}
    [rows, cols] = data.shape
    # print(data[:3])
    valid_tra = []
    for i in tqdm(range(rows)):
        valid_tra.append([lab_key[data[i,0]], data[i,1], data[i,2]])

    data_numpy = np.array(valid_tra)
    data_numpy = pd.DataFrame(data_numpy, index=None, columns=None)
    data_numpy.to_csv('./data/valid_crop_tra.txt', header=None, index=None)
    # print("over")

    crop = pd.read_csv('./data/valid_crop_tra.txt',header=None)
    crop = crop.values

    background = pd.read_csv('./data/final-other-data.txt',header=None)
    background = background.values

    merge_data = np.concatenate((crop,background),axis=0)
    merge_data = pd.DataFrame(merge_data, index=None, columns=None)
    merge_data.to_csv('./data/final_train_data.txt', header=None, index=None)

    ss = "./data/final_train_data.txt"
    data1 = get_tif_data(ss)
    data2 = get_roat_tif_data(ss)  ##将图形旋转90度
    data = np.concatenate((data1,data2),axis=0)
    np.save("./data/final_train_data.npy", data)

    return 


dataset = gdal.Open(
    r"./data/GF6_WFV_E127.9_N46.8_20180823_L1A1119838015.tif")
bands = [i + 1 for i in range(8)]
print(bands)
print(dataset.RasterXSize)
print(dataset.RasterYSize)
labels_key = {'玉米': 0, '大豆': 1, '水稻': 2,'其他': 3}

if __name__ == '__main__':


    ss = "data/sampling-data.txt"
    data1 = get_tif_data_5(ss)
    np.save("data/sampling_data.npy", data1)

    # os.system("mkdir -p model")
    train()
    test()
    create_final_train_data()






