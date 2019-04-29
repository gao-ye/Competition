# -*- coding: utf-8 -*-
#pytorch.__version__ = 0.4.0
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

#from cnn import CNN
from resnet import ResNet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model =ResNet()
model = model.cuda()

import hyperpara as para ##引用参数
image_size = para.image_size  #the size of image to input cnn-net
tra_size   = para.tra_size
val_size   = para.val_size
num_epochs = para.num_epochs
batch_size = para.batch_size
val_batch_size = para.val_batch_size


tra_size = tra_size * 2 
break_batch_tra = int(tra_size/batch_size)
break_batch_val = int(val_size/val_batch_size)

print(break_batch_tra)
print(break_batch_val)

class_num = 4
learning_rate=0.0001

def load_data():
    data = np.load("./data/final_train_data.npy")
    # data = np.load("data/large-train_raw.npy")
    np.random.shuffle(data)
    y= data[:,1]
    return np.array([i for i in data[:,0]]),y

def load_val_data():
    data = np.load("./data/divided-val_raw.npy")
    # data = np.load("data/large-train_raw.npy")
    # data = np.random.shuffle(data)
    np.random.shuffle(data)
    y= data[:,1]
    return np.array([i for i in data[:,0]]),y

def train(x,y):

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

    # x_v = np.asarray(x_v, dtype =  np.int32)  
    # y_v = np.asarray(y_v, dtype =  np.int32)  
    # x_v_torch=torch.from_numpy(x_v)
    # y_v_torch=torch.from_numpy(y_v)

    # # 先转换成 torch 能识别的 Dataset
    # torch_dataset = Data.TensorDataset(x_v_torch, y_v_torch)
    
    # # 把 dataset 放入 DataLoader
    # val_loader = Data.DataLoader(
    #     dataset=torch_dataset,      # torch TensorDataset format
    #     batch_size=batch_size,      # mini batch size
    #     shuffle=False,               # 要不要打乱数据 (打乱比较好)
    #     num_workers=1,              # 多线程来读数据
    # )

    #traing the  model
    for epoch in range(num_epochs):
        model.train()
        print("epoch {}".format(epoch))
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            #print(i)
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
            print("train accuracy is {}".format(accu))

        # if 1 :
        #     model.eval()
        #     total = 0
        #     correct = 0
        #     t00 = 0
        #     t01 = 0
        #     t10 = 0
        #     t11 = 0
        #     t20 = 0
        #     t21 = 0
        #     t30 = 0
        #     t31 = 0


        #     for i, (images, labels) in enumerate(val_loader):
        #         images = Variable(images)
        #         labels = Variable(labels)
      
        #         images=images.float()
        #         labels=labels.long()

        #         if i == break_batch_val :        #进行到最后一个batch_size时, reshape 无法
        #             break                   # 使用，直接跳过
            
        #         labels = labels.reshape(batch_size)      

        #         outputs = model(images.cuda())
        #         pred_data, pred = torch.max(outputs.data,1)		##取出经过网络计算后的标签
                

        #         pred = pred.cpu().numpy()
        #         labels = labels.numpy()

        #         # print(labels)
        #         for i in range(batch_size):
        #             if labels[i] == 0:
        #                 t01 +=1
        #                 if pred[i] == 0 :
        #                     t00 +=1
        #             elif labels[i] == 1:
        #                 t11 +=1
        #                 if pred[i] == 1 :
        #                     t10 +=1
        #             elif labels[i] == 2:
        #                 t21 +=1
        #                 if pred[i] == 2 :
        #                     t20 +=1                           
        #             elif labels[i] == 3:
        #                 t31 +=1
        #                 if pred[i] == 3 :
        #                     t30 +=1


        #         correct += (pred == labels).sum()
        #         total += np.size(labels,0)
        
        #     accu = correct/total
        #     print("val correct is {}".format(correct))
        #     print("val total is {}".format(total))
        #     print("val accuracy is {}".format(accu))
        #     print("kind 0 correct: {} total: {} accu : {}".format(t00,t01, t00/t01))
        #     print("kind 1 correct: {} total: {} accu : {}".format(t10,t11, t10/t11))
        #     print("kind 2 correct: {} total: {} accu : {}".format(t20,t21, t20/t21))
        #     print("kind 3 correct: {} total: {} accu : {}".format(t30,t31, t30/t31))
    #print(test)	
        torch.save(model, 'model/'+str(epoch)+'resnet-model.pkl')
            
    
    torch.save(model, 'model/resnet-model.pkl')
    print("model has been written")
    # model = torch.load('model.pkl')

    return 






if __name__ == '__main__':
    import os
    # os.system("mkdir -p model")
    x,y = load_data()
    # x_val,y_val = load_val_data()
    # print(x.shape)
    # print(y.shape)
    # print(x.shape,y.shape)

    import time

    #begin time
    start = time.clock()
    train(x,y)
    #end time
    end = time.clock()
    second = end-start
    minute = int(second /60)
    second = int(second - minute*60)
    print ("time is  {0} minute {1} second ".format(minute, second))

