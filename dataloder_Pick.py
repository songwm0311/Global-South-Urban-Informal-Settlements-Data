from os.path import join
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
#
# noinspection PyUnboundLocalVariable
class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, namelist, dir, DIRLIST):
        super(DatasetFromFolder, self).__init__()
        self.name_list = namelist
        self.dir = dir
        self.DIRLIST = DIRLIST

    def __getitem__(self, index):
        
        label = sio.loadmat(join (self.dir, self.DIRLIST[1])  + self.name_list[index])
        label = label[list(label.keys())[3]]


        label_new = np.zeros((1,label.shape[0],label.shape[1]),dtype=np.float32)
        label_new[0,:,:] = label
        label_new[np.isnan(label_new)] = 0
        #load the  input

        factor = sio.loadmat(join (self.dir, self.DIRLIST[0]) + self.name_list[index])
        factor = factor[list(factor.keys())[3]]
        factor[np.isnan(factor)] = -999
        factor[factor < 0] = 0
        factor[factor > 1] = 1
        mask = np.where((factor[:, :, 0:1] <= 0), 0, 1)




        factor_Input = np.transpose(factor, (2, 0, 1))
        mask = np.transpose(mask,(2,0,1))

        return factor_Input, label_new, mask

    def __len__(self):
        return len(self.name_list)

