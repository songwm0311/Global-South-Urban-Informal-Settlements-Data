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
# from config_all import * #全部数据的测试
from models.UnetFamily.Unet3PLUS import Unet3Plus
from plt_data import *
import scipy.io as sio
# from osgeo import gdal




def cal_mse(pre,lab,mask):
    pre    = pre[:,0,:,:]*mask
    label  = lab[:,0,:,:]*mask
    dt     = abs(pre-label)
    dt_1   = dt.sum(axis=2)
    mask_1 = mask.sum(axis=2)
    return dt_1,mask_1

def change_device(data):
    #GPU数据转到CPU
    data = data.cpu()
    out = data.detach().numpy()
    return out


os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
model = Unet3Plus.UNet_3Plus(in_channels=13, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
model.load_state_dict(torch.load('/mnt/PRESKY/user/cuijiawen/jm/codes/PT/Unet3Plus/best_model.pt'))##
model.eval()
model = torch.nn.DataParallel(model).cuda()

nn = 1
outpath = '/mnt/PRESKY/user/cuijiawen/jm/codes/out_test/'

filen = '/mnt/PRESKY/user/cuijiawen/jm/Mumbai-all.mat'

data = sio.loadmat(filen)
data = (data[list(data.keys())[3]]).astype(float)
data_nor = (data-500)/8500
data_nor[data_nor<0] = 0
data_nor[data_nor>1] = 1
data_nor[np.isnan(data_nor)] = 0
data_in1 = np.transpose(data_nor, (2, 0, 1))

data_normal = torch.tensor(np.tile(data_in1,(1,1,1,1)))
data_normal = data_normal.type(torch.float32).cuda(non_blocking=True)
outdata = np.zeros((data_normal.shape[2],data_normal.shape[3]),dtype = np.float32)

nl = data_normal.shape[2]
ns = data_normal.shape[3]

for ii in range(int(np.ceil(nl / 128))):
    for jj in range(int(np.ceil(ns / 128))):
        indx1 = ii*128
        indx2 = jj*128
        if ii*128+128>nl:
            indx1 = nl-128
        else:
            indx1 = indx1
        if jj*128+128>ns:
            indx2 = ns-128
        else:
            indx2 = indx2

        data_windows = data_normal[:,:,indx1:indx1+128,indx2:indx2+128]
        outputs = model(data_windows)

        xx = change_device(outputs[0,0,:,:])
        xx[xx>0.65] = 1
        xx[xx<1] = 0

        outdata[indx1:indx1+128,indx2:indx2+128] = xx
outname1 = outpath + 'predict-mumbai.mat'
sio.savemat(outname1, {'predict': outdata})

