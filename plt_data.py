# -*- coding: utf-8 -*-
"""
反归一化，计算RMSE，unit：%
图形绘制：
（1）2合1图：label，predict
（2）test mse图
"""
import numpy as np
import math
import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000



def r_nor(data):
    out = data*0.0
    for i in range(128):
        for j in range(128):
            tt = data[i,j]
            out[i,j] = math.exp(tt*np.log(18)) - 1
    return out

def cal_rmse(data,label):
    label = r_nor(label)
    data  = r_nor(data)
    num = np.sum(data>0)
    ee = (data-label)**2
    rmse = math.sqrt(1.0/num * ee.sum())
    return rmse

def plt_sh700(data,label,outname):
    ##
    
    data[data == 0] = np.nan
    label[label == 0] = np.nan
    title1 = 'ERA5 sh700'
    title2 = 'AMSUB sh700'
    lon_era = np.arange(96., 128., 0.25)
    lat_era = np.arange(16., 48., 0.25)
    lon, lat = np.meshgrid(lon_era, lat_era)
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1)
    ####----------------------------------------------####
    plt.pcolor(lon, lat, np.transpose(label), cmap='Spectral_r', vmin=0, vmax=0.10)  # 绘图，并设置图例两端显示尖端
    cb = plt.colorbar()  # 图例在底端显示
    plt.title(title1, fontsize=12, weight='bold')
    ax = fig.add_subplot(1, 2, 2)
    ####----------------------------------------------####
    plt.pcolor(lon, lat, np.transpose(data), cmap='Spectral_r', vmin=0, vmax=0.10)  # 绘图，并设置图例两端显示尖端
    cb = plt.colorbar()  # 图例在底端显示
    plt.title(title2, fontsize=12, weight='bold')
    plt.savefig(outname,dpi = 600)
    
def plt_mse(data,outname,title_n):
    data[data == 0] = np.nan
    lon_era = np.arange(96., 128., 0.25)
    lat_era = np.arange(16., 48., 0.25)
    lon, lat = np.meshgrid(lon_era, lat_era)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ####----------------------------------------------####
    plt.pcolor(lon, lat, np.transpose(data), cmap='Spectral_r', vmin=np.min(data), vmax=np.max(data))  # 绘图，并设置图例两端显示尖端
    cb = plt.colorbar()  # 图例在底端显示
    plt.title(title_n, fontsize=12, weight='bold')
    plt.savefig(outname, dpi=600)
    
def cal_errors(x,y):
    mse = np.sum((x - y) ** 2) / len(y)
    rmse = math.sqrt(mse)
    mae = np.sum(np.absolute(x-y)) / len(y)
    r2 = 1 - mse / np.var(y)  # 均方误差/方差
    print(" mae:", mae, "mse:", mse, " rmse:", rmse, " r2:", r2)
    return mse,rmse,mae,r2

    
    
    
    