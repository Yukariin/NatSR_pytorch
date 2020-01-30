from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze_(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, full=False):
    pad = window_size//2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        v1 = 2*sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        return ret, cs
    
    return ret


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        c = img1.size(1)

        if c == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, c)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = c

        return _ssim(img1, img2, window, self.window_size, c, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True, full=False):
    _, c, h, w = img1.size()

    real_size = min(window_size, h, w)
    window = create_window(real_size, c)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, real_size, c, size_average, full)


def msssim(img1, img2, window_size=11, size_average=True):
    weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    if img1.is_cuda:
        weights = weights.cuda(img1.get_device())
    weights = weights.type_as(img1)

    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size, size_average, full=True)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


def psnr(input, target):
    mse = F.mse_loss(input, target)
    psnr = 10 * torch.log10(1. / mse)

    return psnr


def calc_acc(input, target):
    eq = torch.eq(torch.gt(input, 0.5).float(), target)
    return 100 * torch.mean(eq.float())
