import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torch
import numpy as np

def plt_bt(data1,data2,name):
    
    data1 = data1[0,0,:,:]
    data2 = data2[0,0,:,:]
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文

    plt.subplot(1, 2, 1)
    plt.title('AI-SH700')
    plt.imshow(data1, cmap='RdBu')
    plt.colorbar(extend='both', label='noisy points extend')
    plt.clim(0, 10)

    plt.subplot(1, 2, 2)
    plt.title('ERA5-SH700')
    plt.imshow(data2, cmap='RdBu')
    plt.colorbar(extend='both', label='noisy points extend')
    plt.clim(0, 10)

    plt.tight_layout()
    #plt.show()
    plt.savefig(name)
# 绘图



