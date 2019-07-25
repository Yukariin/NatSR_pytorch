import numpy as np
import numpy.random as npr
from scipy.fftpack import dct, idct
import torch
import torch.nn.functional as F


def inject_dct(x, sigma):
    n, c, h, w = x.shape
    X_space = np.reshape(x, [n, c, h//8, 8, w//8, 8])
    X_dct_x = dct(X_space, axis=3, norm='ortho')
    X_dct = dct(X_dct_x, axis=5, norm='ortho')

    noise_raw = npr.randn(n, c, h//8, 8, w//8, 8) * sigma
    z = np.zeros([n, c, h//8, 8, w//8, 8])
    z[:, :, :, 7, :, :] = noise_raw[:, :, :, 7, :, :]
    z[:, :, :, :, :, 7] = noise_raw[:, :, :, :, :, 7]

    X_dct_noise = X_dct + z

    Y_space_x = idct(X_dct_noise, axis=3, norm='ortho')
    Y_space = idct(Y_space_x, axis=5, norm='ortho')
    Y = np.reshape(Y_space, x.shape)

    return Y


def get_manifold(x, scale=4, alpha=0.5, sigma=0.1):
    lr = F.interpolate(x, scale_factor=1/scale, mode='bicubic')
    lr = F.interpolate(lr, scale_factor=scale, mode='bicubic')
    a = (1-alpha)*lr + alpha*x

    b = inject_dct(x.cpu().numpy(), sigma)
    b = torch.from_numpy(b).float().to(x.device)
    return a, b
