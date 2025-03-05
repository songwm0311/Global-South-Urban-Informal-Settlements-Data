import torch
import argparse
import os
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset, random_split
from os.path import join
from os import listdir
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
# import time
from config_all_pj import test_loader
# from models.UnetFamily.Unet import UNet
# from models.UnetFamily.Unet2PLUS import UNet2plus
from models.UnetFamily.Unet3PLUS import Unet3Plus
from plt_data import *
# from models.UnetFamily.U2NET import u2net
# from models.UnetFamily.MSUNET import msunet

def cal_mse(pre,lab,mask):
    pre    = pre[:,0,:,:]*mask
    label  = label[:,0,:,:]*mask
    dt     = abs(pre-label)
    dt_1   = dt.sum(axis=2)
    mask_1 = mask.sum(axis=2)
    return dt_1,mask_1

def r_nor(data):#反归一化
    out = data*0.0
    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                tt = data[k,i,j]
                out[k,i,j] = math.exp(tt*np.log(18)) - 1
    return out

def change_device(data):
    #GPU数据转到CPU
    data = data.cpu()
    out = data.detach().numpy()
    return out

def cal_re(data,label,mask):#relative error--->abs(A-B)/B
    #计算相对误差
    data  = r_nor(data[:,0,:,:])
    label = r_nor(label[:,0,:,:])
    mask  = mask[:,0,:,:]
    re  = (abs(data-label)/label)*mask
    return re.sum(axis=0),mask.sum(axis=0)

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
model = Unet3Plus.UNet_3Plus(in_channels=5, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
# model.load_state_dict(torch.load('/mnt/PRESKY/user/cuijiawen/TT/PT/Unet3Plus/ssimloss0905_band1to5_0908.pt'))
model.load_state_dict(torch.load('/mnt/PRESKY/user/cuijiawen/TT/PT/Unet3Plus/mseloss1011_band5.pt'))
model.eval()
model = torch.nn.DataParallel(model).cuda()  # 多卡运行

outpath = '/mnt/PRESKY/user/cuijiawen/predata/pj/world'
loader = test_loader

for batch_idx, (factor_Input,  humidity, mask) in enumerate(tqdm(loader)):
    factor_Input,  humidity,  mask = factor_Input.float().cuda(non_blocking=True), \
                                                          humidity.type(torch.float32).cuda(non_blocking=True), \
                                                          mask.cuda(non_blocking=True)
    print(batch_idx)
    outputs = model(factor_Input)
    y_p   = change_device(outputs[:,0,:,:])   ##3D
    y_r   = change_device(humidity[:,0,:,:])  ##3D
    mask1 = change_device(mask[:,0,:,:])      ##3D
    ff = change_device(factor_Input)
    y_p = r_nor(y_p)                          ##反归一化predict
    y_r = r_nor(y_r)                          ##反归一化label
    # y_p[mask1 == 0] = np.nan
    # y_r[mask1 == 0] = np.nan

    # outname1 = outpath + '/bt1-' + str(batch_idx+1) + '.mat'
    # outname2 = outpath + '/predict1-' + str(batch_idx+1) + '.mat'
    # outname3 = outpath + '/mask1-' + str(batch_idx+1) + '.mat'
    #
    # sio.savemat(outname1, {'bt': ff})
    # sio.savemat(outname2, {'predict': y_p})
    # sio.savemat(outname3, {'mask': mask1})




