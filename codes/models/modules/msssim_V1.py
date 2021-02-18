# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y,
          data_range,
          win,
          size_average=True,
          K=(0.01, 0.03)):
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs

def ms_ssim(X, Y,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            win=None,
            weights=None,
            K=(0.01, 0.03)):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same dimensions.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y,
                                     win=win,
                                     data_range=data_range,
                                     size_average=False,
                                     K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = (X.shape[2] % 2, X.shape[3] % 2)
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)

def gauss_weighted_l1(X, Y, win=None, win_size=33, win_sigma=8, size_average=False):
    diff = abs(X - Y)
    (_, channel, height, width) = diff.size()
    if win is None:
        real_size = min(win_size, height, width)
        win = _fspecial_gauss_1d(real_size, win_sigma)
        win = win.repeat(diff.shape[1], 1, 1, 1)

    win = win.to(diff.device, dtype=diff.dtype)
    l1 = gaussian_filter(diff, win)
    if size_average:
        return l1.mean()
    else:
        return l1
def ms_ssim_l1_loss(X, Y, data_range, size_average, alpha, win_l1, win_ssim, weights, K, mean_metric):
    ms_ssim_map = ms_ssim(X, Y, data_range, size_average, win_ssim, weights, K)
    l1_map = gauss_weighted_l1(X, Y, win_l1, size_average)
    print('ms_ssim_map.shape')
    print(ms_ssim_map.shape)
    print('l1_map.shape')
    print(l1_map.shape)
    loss_map = (1 - ms_ssim_map) * alpha + l1_map * (1 - alpha)
    if mean_metric:
        return loss_map.mean()
    else:
        return loss_map

class MSSSIML1_Loss(torch.nn.Module):
    def __init__(self,
                 data_range=None,
                 size_average=False,
                 alpha=0.84,
                 win_size_l1=33,
                 win_sigma_l1=8,
                 win_size_ssim=11,
                 win_sigma_ssim=1.5,
                 channel=1,
                 weights=None,
                 K=(0.01, 0.03),
                 mean_metric=True):

        super(MSSSIML1_Loss, self).__init__()
        self.win_size_l1 = win_size_l1
        self.win_sigma_l1 = win_sigma_l1
        self.win_l1 = _fspecial_gauss_1d(win_size_l1, win_sigma_l1).repeat(channel, 1, 1, 1)
        self.win_size_ssim = win_size_ssim
        self.win_sigma_ssim = win_sigma_ssim
        self.win_ssim = _fspecial_gauss_1d(win_size_ssim, win_sigma_ssim).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.alpha = alpha
        self.K = K
        self.mean_metric = mean_metric
    def forward(self, X, Y):
        if not X.shape == Y.shape:
            raise ValueError('Input images must have the same dimensions.')
        return ms_ssim_l1_loss(X, Y,
                               data_range=self.data_range,
                               size_average=self.size_average,
                               alpha=self.alpha,
                               win_l1=self.win_l1,
                               win_ssim=self.win_ssim,
                               weights=self.weights,
                               K=self.K,
                               mean_metric=self.mean_metric)