import copy

import numpy as np
import numpy.random as npr
from scipy.fftpack import dct, idct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ImageSplitter:
    def __init__(self, seg_size=48, scale=4, pad_size=3):
        self.seg_size = seg_size
        self.scale = scale
        self.pad_size = pad_size
        self.height = 0
        self.width = 0
        self.ref_pad = nn.ReplicationPad2d(self.pad_size)

    def split(self, pil_img):
        img_tensor = TF.to_tensor(pil_img).unsqueeze_(0)
        img_tensor = self.ref_pad(img_tensor)
        _, _, h, w = img_tensor.size()
        self.height = h
        self.width = w

        if h % self.seg_size < self.pad_size or w % self.seg_size < self.pad_size:
            self.seg_size += self.scale * self.pad_size

        patches = []
        for i in range(self.pad_size, h, self.seg_size):
            for j in range(self.pad_size, w, self.seg_size):
                patch = img_tensor[:, :,
                    (i-self.pad_size):min(i+self.pad_size+self.seg_size, h),
                    (j-self.pad_size):min(j+self.pad_size+self.seg_size, w)]

                patches.append(patch)

        return patches

    def merge(self, patches):
        pad_size = self.scale * self.pad_size
        seg_size = self.scale * self.seg_size
        height = self.scale * self.height
        width = self.scale * self.width

        out = torch.zeros((1, 3, height, width))
        patch_tensors = copy.copy(patches)

        for i in range(pad_size, height, seg_size):
            for j in range(pad_size, width, seg_size):
                patch = patch_tensors.pop(0)
                patch = patch[:, :, pad_size:-pad_size, pad_size:-pad_size]

                _, _, h, w = patch.size()
                out[:, :, i:i+h, j:j+w] = patch
        out = out[:, :, pad_size:-pad_size, pad_size:-pad_size]

        return TF.to_pil_image(out.clamp_(0,1).squeeze_(0))


def inject_dct(x, sigma):
    b, c, h, w = x.shape
    X_space = np.reshape(x, [b, c, h//8, 8, w//8, 8])
    X_dct_x = dct(X_space, axis=3, norm='ortho')
    X_dct = dct(X_dct_x, axis=5, norm='ortho')

    noise_raw = npr.randn(b, c, h//8, 8, w//8, 8) * sigma
    z = np.zeros([b, c, h//8, 8, w//8, 8])
    z[:, :, :, 7, :, :] = noise_raw[:, :, :, 7, :, :]
    z[:, :, :, :, :, 7] = noise_raw[:, :, :, :, :, 7]

    X_dct_noise = X_dct + z

    Y_space_x = idct(X_dct_noise, axis=3, norm='ortho')
    Y_space = idct(Y_space_x, axis=5, norm='ortho')
    Y = np.reshape(Y_space, x.shape)

    return Y


def get_noisy(x, sigma):
    b = inject_dct(x.cpu().numpy(), sigma)
    b = torch.from_numpy(b).float().to(x.device)
    return b


def get_blurry(x, scale, alpha):
    lr = F.interpolate(x, scale_factor=1/scale, mode='bicubic')
    lr = F.interpolate(lr, scale_factor=scale, mode='bicubic')
    a = (1-alpha)*lr + alpha*x

    return a
