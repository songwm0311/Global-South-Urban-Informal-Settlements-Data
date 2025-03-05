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

filen = '/mnt/PRESKY/user/cuijiawen/jm/testdata.mat'

data = sio.loadmat(filen)
data = (data[list(data.keys())[3]]).astype(float)
# data_nor = data
data_nor = (data-500)/8500
data_nor[data_nor<0] = 0
data_nor[data_nor>1] = 1
data_nor1 = np.zeros((data_nor.shape[0]+40,data_nor.shape[1]+40,data_nor.shape[2]),dtype = np.float32)
data_nor1[20:20+data_nor.shape[0],20:20+data_nor.shape[1],:] = data_nor
data_nor1[0:20,20:20+data_nor.shape[1],:] = np.flipud(data_nor[0:20,:,:])
data_nor1[data_nor.shape[0]:data_nor.shape[0]+20,20:20+data_nor.shape[1],:] = np.flipud(data_nor[data_nor.shape[1]-20:data_nor.shape[0],:,:])
data_nor1[:,0:20,:] = np.fliplr(data_nor1[:,0:20,:])
data_nor1[:,data_nor.shape[1]-20:data_nor.shape[1],:] = np.fliplr(data_nor1[:,data_nor.shape[1]-20:data_nor.shape[1],:])




data_in1 = np.transpose(data_nor1, (2, 0, 1))
# sio.savemat('/mnt/PRESKY/user/cuijiawen/jm/codes/tt.mat', {'in': data_in1})

data_normal = torch.tensor(np.tile(data_in1,(1,1,1,1)))
data_normal = data_normal.type(torch.float32).cuda(non_blocking=True)
outdata = np.zeros((data_normal.shape[2],data_normal.shape[3]),dtype = np.float32)

num_nl = int(np.ceil(data_normal.shape[2]/88))
num_ns = int(np.ceil(data_normal.shape[3]/88))

for ii in range(num_nl):
    for jj in range(num_ns):
        nl = ii*88
        ns = jj*88
        if nl+128>data_normal.shape[2]:
            nl = data_normal.shape[2]-128
        else:
            nl = nl

        if ns+128>data_normal.shape[3]:
            ns = data_normal.shape[3]-128
        else:
            ns = ns

        print(nl,ns)
        data_windows = data_normal[:,:,nl:nl+128,ns:ns+128]
        outputs = model(data_windows)
        xx = change_device(outputs[0,0,:,:])
        xx[xx>0.65] = 1
        xx[xx<1] = 0
        xx[0:20,:] = 0
        xx[108:128,:] = 0
        xx[:,0:20] = 0
        xx[:,108:128] = 0
        tt = np.zeros(outdata.shape,dtype = np.float32)
        tt[nl:nl+128,ns:ns+128] = xx
        outdata = outdata + tt
nl1 = outdata.shape[0]
ns1 = outdata.shape[1]
outname1 = outpath + 'predict-test4.mat'
sio.savemat(outname1, {'predict': outdata[20:nl1-20,20:ns1-20]})





# predict = np.zeros((data.shape[0],data.shpe[1]),dtype=np.float32)
# for

#
#
#
# for batch_idx, (factor_Input, humidity, mask) in enumerate(tqdm(loader)):
#     factor_Input,  humidity,  mask = factor_Input.float().cuda(non_blocking=True), \
#                                                           humidity.type(torch.float32).cuda(non_blocking=True), \
#                                                           mask.cuda(non_blocking=True)
#     print(factor_Input.shape)
#     outputs = model(factor_Input)
#     xx = change_device(factor_Input)
#     # print(factor_Input.shape)## size = batch_szie*13*128*128
#     # print(outputs.shape)     ## size = batch_szie*1*128*128
#     # print(humidity.shape)  ## size = batch_szie*1*128*128
#
#     y_p   = change_device(outputs[:,0,:,:])   ##3D
#     y_r   = change_device(humidity[:,0,:,:])  ##3D
#     y_r[y_r>0.5] = 1
#     y_r[y_r<1] = 0
#     mask1 = change_device(mask[:,0,:,:])      ##3D
#
#     for ii in range(y_p.shape[0]):
#         outname1 = outpath + 'predict-' + str(nn).zfill(3) + '.mat'
#         sio.savemat(outname1, {'predict': y_p[ii,:,:]})
#         outname2 = outpath + 'label-' + str(nn).zfill(3) + '.mat'
#         sio.savemat(outname2, {'label': y_r[ii,:,:]})
#         outname3 = outpath + 'mask-' + str(nn).zfill(3) + '.mat'
#         sio.savemat(outname3, {'mask': mask1[ii,:,:]})
#         nn = nn+1


#
#     y_p = r_nor(y_p)                          ##反归一化predict
#     y_r = r_nor(y_r)                          ##反归一化label
#     y_p[mask1 == 0] = 0
#     y_r[mask1 == 0] = 0
#     outpath = '/mnt/PRESKY/user/cuijiawen/paper/'
#
#     # for i in range(y_p.shape[0]):
#     #     output = mhs_names_test[nn]
#     #     outname1 = outpath + output
#     #     sio.savemat(outname1, {'sh850': y_p[i]})
#     #     nn = nn+1
#
# # #计算RMSE,MSE,MAE等系列指标
#     pp_all = np.concatenate([pp_all, y_p], axis=0)
#     ll_all = np.concatenate([ll_all, y_r], axis=0)
#     mm_all = np.concatenate([mm_all, mask1], axis=0)
# # ###############---------------------------------------------################
# #     ##统计原始数据的区域均值##
#     predict_sum = predict_sum + y_p.sum(axis=0)
#     label_sum = label_sum + y_r.sum(axis=0)
#     mask_sum = mask_sum + mask1.sum(axis=0)
# #
#     dt = (abs(y_p - y_r)).sum(axis = 0)
#     dt_sum = dt_sum + dt
#     # mask_sum = mask_sum + mask1.sum(axis = 0)
#
# mask_sum[np.where(mask_sum==0)] = 1
# error = dt_sum/mask_sum
#
#
#
# p_out = predict_sum/mask_sum
# l_out = label_sum/mask_sum
# ##########################   结果保存   #####################################
# outname1 = outpath + '700predict_mean.mat'
# sio.savemat(outname1, {'p_m': p_out})
# outname2 = outpath + '700label_mean.mat'
# sio.savemat(outname2, {'l_m': l_out})
# outname3 = outpath + '700mask_mean.mat'
# sio.savemat(outname3, {'m_m': mask_sum})
# outname4 = outpath + 'rh700_abs_error.mat'
# sio.savemat(outname4, {'error': error})
#
# # outname2 = outpath + 'label_mean.mat'
# # sio.savemat(outname2, {'label': l_out})
#
#
# # plt_mse(dt_out/100.0, outname, 'ERROR')
# ####################计算误差指标#######################
# pp_all[mm_all==0] = -999
# ll_all[mm_all==0] = -999
# pp_1 = pp_all.flatten()
# ll_1 = ll_all.flatten()
#
# pp_1 = pp_1[pp_1 != -999]
# ll_1 = ll_1[ll_1 != -999]
# mse, rmse, mae, r2 = cal_errors(pp_1, ll_1)
