from __future__ import print_function
from tensorboardX import SummaryWriter
import argparse
from torchsummary import summary
import os
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset, random_split
from os.path import join
from os import listdir
import shutil
import time
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from loss import LossSelector
from utils import AverageMeter
from torchsummary import summary
from tqdm import tqdm
import time
from config_all import train_loader,val_loader,test_loader
from models.UnetFamily.Unet import UNet
from models.UnetFamily.Unet2PLUS import UNet2plus
from models.UnetFamily.Unet3PLUS import Unet3Plus
from models.UnetFamily.U2NET import u2net
from models.UnetFamily.MSUNET import msunet
from PLT_imshow import plt_bt
import scipy.io as sio
########################################################################################################################
##################################################Model Cofig###########################################################
########################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,4'


mseloss_params = {'weight': None}
msssimloss_params = {'weight': None}
l1loss_params = {'weight': None}
weightmseloss_params = {'weight': None}



parser = argparse.ArgumentParser(description='PyTorch AV Training')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
# parser.add_argument('--t', '--t', default=16, type=int,  help='initial learning rate')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--val_iteration', type=int, default=1, help='Number of labeled data')
parser.add_argument('--out', default='result', help='Directory to output the result')
parser.add_argument('--model', default='Unet3Plus',
                    choices=['UNet','UNet2plus','Unet3Plus','UNet_3Plus_DeepSup','UNet_3Plus_DeepSup_CGM','u2net','U2NETP','msunet'],
                    type=str)
parser.add_argument('--loss', default='mseloss',
                    choices=['l1loss', 'mseloss', 'weightmseloss', 'logcoshloss','msssimloss'], type=str)
parser.add_argument('--loss_params', default={'l1loss': l1loss_params,
                                              'mseloss': mseloss_params,
                                              'weightmseloss': weightmseloss_params,
                                              'msssimloss':msssimloss_params})
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  # for numpy
    torch.manual_seed(seed)  # for CPU
    torch.cuda.manual_seed(seed)  # for current GPU
    torch.cuda.manual_seed_all(seed)  # for all GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


setup_seed(42)

cudnn.benchmark = True

PATH = args.model
start_epoch = 0
inch = 20
class_num = 1

reduction_ratio = 16
kernel_layer = 2


best_dice = 10.





def main():
    global best_dice
    ################################################## Initialize network###############################################

    if args.model == 'UNet':
        model = UNet.UNet(in_channels=13, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    elif args.model == 'UNet2plus':
        model = UNet2plus.UNet_2Plus(in_channels=13, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True)
    elif args.model == 'Unet3Plus':
        model = Unet3Plus.UNet_3Plus(in_channels=13, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    elif args.model == 'UNet_3Plus_DeepSup':
        model =  Unet3Plus.UNet_3Plus_DeepSup(in_channels=13, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True,other=False)
    elif args.model == 'UNet_3Plus_DeepSup_CGM':
        model = Unet3Plus.UNet_3Plus_DeepSup_CGM(in_channels=13, n_classes=1, feature_scale=4, is_deconv=True,is_batchnorm=True, other=False)
    elif args.model  == 'u2net':
        model = u2net.U2NET(in_ch=13, out_ch=1, other=False)
    elif args.model == 'U2NETP':
        model = u2net.U2NETP(in_ch=13, out_ch=1, other=False)
    elif args.model == 'msunet':
        model = msunet.U_Net(img_ch=13, output_ch=1)

    model = torch.nn.DataParallel(model).cuda()  # 多卡运行

    print('model', model)
    print('loss', args.model, args.loss)

    ###########Initialize loss##########
    criterion = LossSelector[args.loss](**args.loss_params[args.loss])

    ###########Calculate network parameters#########
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    # StepLR = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

    #####自动创建保存模型的目录文件夹############
    model_fold = "./PT/" + PATH + '/'
    os.makedirs(model_fold, exist_ok=True)

    writer = SummaryWriter()


    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        since = time.time()
        epo = epoch+1

        train_losses, train_err ,train_rmse = Driver(train_loader, model, optimizer, criterion,epo, TRAIN=True, Tips='TRAIN')
        val_losses, val_err, val_rmse= Driver(val_loader, model, optimizer, criterion, epo,TRAIN=False, Tips='VAL')
        test_losses, test_err, test_rmse = Driver(test_loader, model, optimizer, criterion, epo,TRAIN=False, Tips='TEST')


        #########统计时间##########
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


        ### 保存权重策略

                                                                                             # 按哪个指标保存
        if val_err < best_dice:
            best_dice = min(val_err, best_dice)
            name = model_fold + 'best_model.pt'
            torch.save(model.module.state_dict(), name)

        # #loss训练验证测试
        writer.add_scalars('LOSS', {'Train_loss': train_losses, 'Val_loss': val_losses,
                                                                 'Test_loss': test_losses},epoch)

        writer.add_scalars('绝对值误差', {'train_err': train_err, 'val_err': val_err,
                                                                 'test_err': test_err},epoch)

        writer.add_scalars('RMSE', {'train_rmse': train_rmse, 'val_rmse': val_rmse,
                                     'test_rmse': test_rmse}, epoch)




    writer.close()

def Driver(loader, model, optimizer, criterion, epo,TRAIN=True,Tips='ERA5 TRAIN'):
    losses = AverageMeter()
    err = AverageMeter()
    rmses = AverageMeter()

    if TRAIN:
        model.train()
    else:
        model.eval()

    for batch_idx, (factor_Input, humidity, mask) in enumerate(tqdm(loader)):
        
        factor_Input,  humidity,  mask = factor_Input.float().cuda(non_blocking=True), \
                                                          humidity.type(torch.float32).cuda(non_blocking=True), \
                                                          mask.cuda(non_blocking=True)
        
        
        
        outputs = model(factor_Input)
        loss = criterion(outputs, humidity, mask)
        losses.update(loss.item(), factor_Input.size(0))
        ###
        ee = torch.abs(outputs - humidity) * mask
        ERROR = torch.sum(ee) / (mask.sum())
        err.update(ERROR.item(),factor_Input.size(0))
        ###精度评价
        rmse2 = torch.sum(ee * ee) / (mask.sum())
        rmse = torch.sqrt(rmse2)
        rmses.update(rmse.item(),factor_Input.size(0))
        ###
        if TRAIN:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(Tips + '---Loss: {loss:.4f} | ERROR: {err:.4f} | RMSE: {rmse:.4f}'. \
            format(loss=losses.avg, err=err.avg, rmse = rmses.avg))

    return (losses.avg, err.avg, rmses.avg)



if __name__ == '__main__':
    main()
