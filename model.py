import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SN


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out


class RDBlock(nn.Module):
    def __init__(self, in_channels, num_dense, dense_out):
        super().__init__()

        out_channels = in_channels
        layers = []
        for i in range(num_dense):
            layers.append(DenseBlock(out_channels, dense_out))
            out_channels += dense_out
        self.dense = nn.Sequential(*layers)

        self.conv_fusion = nn.Conv2d(out_channels, in_channels, kernel_size=1, padding=0, bias=False)
    
    def forward(self, x):
        out = self.dense(x)
        out = self.conv_fusion(out)
        out = out + x
        return out


class NSRNet(nn.Module):
    def __init__(self, c_img=3, n_feat=64, num_dense=5, dense_out=64, scale=2):
        super().__init__()

        self.conv1 = nn.Conv2d(c_img, n_feat, kernel_size=3, padding=1, bias=True)

        self.block1 = RDBlock(n_feat, num_dense, dense_out)
        self.block2 = RDBlock(n_feat, num_dense, dense_out)

        self.block3 = RDBlock(n_feat, num_dense, dense_out)
        self.block4 = RDBlock(n_feat, num_dense, dense_out)

        self.block5 = RDBlock(n_feat, num_dense, dense_out)
        self.block6 = RDBlock(n_feat, num_dense, dense_out)

        self.block7 = RDBlock(n_feat, num_dense, dense_out)
        self.block8 = RDBlock(n_feat, num_dense, dense_out)

        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True)

        self.conv_up1 = nn.Conv2d(n_feat, n_feat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.conv_up2 = nn.Conv2d(n_feat, n_feat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.sub_pixel = nn.PixelShuffle(scale)

        self.conv_out = nn.Conv2d(n_feat, c_img, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out_1 = self.conv1(x)

        rdb_out_1 = self.block1(out_1)
        rdb_out_1 = self.block2(rdb_out_1)
        rdb_out_1 = rdb_out_1 + out_1

        rdb_out_2 = self.block3(rdb_out_1)
        rdb_out_2 = self.block4(rdb_out_2)
        rdb_out_2 = rdb_out_2 + rdb_out_1 + out_1

        rdb_out_3 = self.block5(rdb_out_2)
        rdb_out_3 = self.block6(rdb_out_3)
        rdb_out_3 = rdb_out_3 + rdb_out_2

        rdb_out_4 = self.block7(rdb_out_3)
        rdb_out_4 = self.block8(rdb_out_4)
        rdb_out_4 = rdb_out_4 + rdb_out_3 + rdb_out_2 + out_1

        out_2 = self.conv2(rdb_out_4)
        out_2 = out_2 + out_1

        up_out1 = self.conv_up1(out_2)
        up_out1 = self.sub_pixel(up_out1)
        up_out2 = self.conv_up2(up_out1)
        up_out2 = self.sub_pixel(up_out2)

        out = self.conv_out(up_out2)

        return out


class Discriminator(nn.Module):
    def __init__(self, c_img=3):
        super().__init__()
        
        cnum = 64
        self.discriminator = nn.Sequential(
            SN(nn.Conv2d(c_img, cnum, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SN(nn.Conv2d(cnum, cnum, 3, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            SN(nn.Conv2d(cnum, cnum*2, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SN(nn.Conv2d(cnum*2, cnum*2, 3, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            SN(nn.Conv2d(cnum*2, cnum*4, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SN(nn.Conv2d(cnum*4, cnum*4, 3, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            SN(nn.Conv2d(cnum*4, cnum*8, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SN(nn.Conv2d(cnum*8, cnum*8, 3, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            SN(nn.Conv2d(cnum*8, cnum*16, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SN(nn.Conv2d(cnum*16, cnum*16, 3, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            SN(nn.Conv2d(cnum*16, 1, 3, 1, 1)),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out


class NMDiscriminator(nn.Module):
    def __init__(self, c_img=3):
        super().__init__()
        
        cnum = 64
        self.discriminator = nn.Sequential(
            nn.Conv2d(c_img, cnum, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnum, cnum, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(cnum, cnum*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnum*2, cnum*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnum*2, cnum*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnum*4, cnum*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnum*4, cnum*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnum*8, cnum*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnum*8, 1, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out
