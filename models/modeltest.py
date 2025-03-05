import torch
from TONGH.models.UnetFamily.Unet import UNet
from TONGH.models.UnetFamily.U2NET import u2net
from TONGH.models.UnetFamily.Unet3PLUS import Unet3Plus
from  TONGH.models.UnetFamily.MSUNET import msunet
from torchsummary import summary
def main():

    in_batch, inchannel, in_h, in_w = 4, 5, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w).cuda()
    net= msunet.U_Net(img_ch=5, output_ch=1).cuda()
    out = net(x)
    summary(net, (5, 128, 128))
    print(net)
    # for i in range(psi1.shape[0]):
    #     print(psi1[i][0].detach().cpu().unsqueeze(dim=0).shape)
    print('输出：',out.shape)

if __name__ == '__main__':
   main()