
from dataloder_Pick import *
import scipy.io as sio
from os.path import join
from os import listdir
#该路径下所有文件名

x_dir = '/mnt/PRESKY/user/cuijiawen/jm/x_train/'
all_names = sorted([x for x in listdir(x_dir) if x.endswith(".mat")])

print("###############################################################################################################################")
train_namelist = all_names[0:1000]
val_namelist = all_names[1000:1200]
test_namelist = all_names[1200:1400]

dir = '/mnt/PRESKY/user/cuijiawen/jm/'
DIRLIST = ['x_train/','y_train/']

datas_train = DatasetFromFolder(train_namelist, dir, DIRLIST)
train_loader = DataLoader(datas_train, batch_size=4, shuffle=True, num_workers=24, pin_memory=True,
                          persistent_workers=True)

datas_val = DatasetFromFolder(val_namelist, dir, DIRLIST)
val_loader = DataLoader(datas_val, batch_size=4, shuffle=False, num_workers=24, pin_memory=True,
                        persistent_workers=True)

test_val = DatasetFromFolder(test_namelist, dir, DIRLIST)
test_loader = DataLoader(test_val, batch_size=4, shuffle=False, num_workers=24, pin_memory=True,
                         persistent_workers=True)


