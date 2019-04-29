import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import multiprocessing
import gc
import sys

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

def make_coord():  ##生成分割格式
    Xsize = 50362
    Ysize = 17810
    step = 1000

    normal_step =1000

    x_s_step = Xsize % step
    y_s_step = Ysize % step

    x_break = int (Xsize / step)
    y_break = int (Ysize / step)

    # print("s_step {} {}".format(x_s_step, y_s_step))
    # print("break {} {}".format(x_break, y_break))
    x = 0
    y = 0
    x_batch = 0
    y_batch = 0
    pic = 0
    coord = []

    while (x < Xsize):
        # print("x_batch is {}".format(x_batch))

        if x_batch == x_break:
            x_step = x_s_step
        else:
            x_step = normal_step

        y_batch = 0
        y       = 0

        while (y < Ysize):
            # print('x_batch is {} y_batch is {}'.format(x_batch, y_batch))
            if y_batch == y_break:
                y_step = y_s_step
            else :
                y_step = normal_step

            coord.append([pic, y, x, y_step, x_step])
            # print(' y-{}, x-{}, y_step-{}, x_step-{}'.format(y, x, y_step, x_step))
            pic +=1
            y_batch +=1
            y += y_step

        x_batch += 1
        x += x_step
    coord = np.array(coord)
    return coord


def segment_pic(s, src, coord):  ##将一张大图分割为若干张小图，分割格式已经记录在 coord中
    [rows, cols] = coord.shape
    for i in tqdm(range(rows)):
        y      = coord[i,1]
        x      = coord[i,2]
        y_size = coord[i, 3]
        x_size = coord[i, 4]

        ss = s +'/'+str(i)+'.png'
        dst = src[y:y+y_size, x:x+x_size]
        cv2.imwrite(ss, dst)

def my_crf_1(fn_im, fn_anno, fn_output):

    img = cv2.imread(fn_im)

    anno_rgb = cv2.imread(fn_anno).astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    print(fn_im)
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    ###########################
    ### Setup the CRF model ###
    ###########################
    use_2d = False
    # use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                            compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                        img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)

    MAP = np.argmax(Q, axis=0)
    MAP = colorize[MAP,:]
    cv2.imwrite(fn_output, MAP.reshape(img.shape))

    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)

def my_crf(begin, end):
    i = begin
    while(i< end):
        try:
            my_crf_1('./code/seg_pic/'+str(i)+'.png', './code/seg_anno/'+str(i)+'.png', './code/res_anno/'+str(i)+'.png')   
        # finally:
        #     i += 1
        except :
            i +=1
        else: 
            i +=1

def merge_pic(coord,src,n):
    for i in tqdm(range(n)):
        ss = './code/res_anno/'+str(i)+'.png'
        mat = cv2.imread(ss, 0)

        if mat is not None:

            y      = coord[i,1]
            x      = coord[i,2]
            y_size = coord[i, 3]
            x_size = coord[i, 4]
            # print(i)
            # print(mat.shape)
            src[y:y+y_size, x:x+x_size] = mat
    return src

if __name__== '__main__':
    os.system('mkdir -p ./code/seg_pic')
    os.system('mkdir -p ./code/seg_anno')
    os.system('mkdir -p ./code/res_anno')


    RasterXSize = 50362
    RasterYSize = 17810

    coord = make_coord()      
    print('make coord over')

    s = './code/seg_pic'
    src = cv2.imread("./data/src.jpg")
    segment_pic(s, src, coord)
    print('segment src.jpg over')

    data = cv2.imread('./code/result/test_result.tif', 0)
    data[data==0]=80
    # print(data[:100,:10])
    print('transform 0 to 80 ok')

    s = './code/seg_anno'
    segment_pic(s, data, coord)  ##将标记大图 分割为小图
    print('make small pic over')

    t1 = multiprocessing.Process(target=my_crf, args=(0,200))
    print('process start')
    t1.start()

    t2 = multiprocessing.Process(target=my_crf, args=(200,400))
    print('process start')
    t2.start()

    t3 = multiprocessing.Process(target=my_crf, args=(400,600))
    print('process start')
    t3.start()

    t4 = multiprocessing.Process(target=my_crf, args=(600,800))
    print('process start')
    t4.start()

    t5 = multiprocessing.Process(target=my_crf, args=(800,918))
    print('process start')
    t5.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()

    result = np.zeros(
    shape=(
        RasterYSize,
        RasterXSize),
    dtype=np.uint8)

    n = 918
    result = merge_pic(coord,result, n)
    print("merge ok")

    result[result == 80] = 0
    # print(result[:100,:10])
    print('transform 80 to 0 over')

    # os.system('mkdir -p final-result')
    cv2.imwrite('./result/test_result.tif', result)
    # cv2.imwrite('final-result/test_result.jpg', result)
    print("all over over")
