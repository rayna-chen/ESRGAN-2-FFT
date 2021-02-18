import torch
import torch.nn.functional as F
from math import exp
import torch.nn as nn


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, sigma=1.5, channel=1):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_tensor(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # cs = torch.mean(v1 / v2)  # contrast sensitivity
    cs_map = v1 / v2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # if size_average:
    #     ret = ssim_map.mean()
    # else:
    #     ret = ssim_map.mean(1).mean(1).mean(1)
    #
    # if full:
    #     return ret, cs
    # return ret
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None, mean_metric=True):
    smaller_side = min(img1.shape[-2:])
    assert smaller_side > (window_size - 1) * (2 ** 4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((window_size - 1) * (2 ** 4))
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim_tensor(img1, img2, window_size=window_size, size_average=False, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    if mean_metric:
        output = output.mean()
    return output

def gauss_weighted_l1(img1, img2, window_size=33, window=None, mean_metric=True):
    diff = abs(img1 - img2)
    (_, channel, height, width) = diff.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, sigma=0.8, channel=channel).to(diff.device)
    padd = 0
    l1 = F.conv2d(diff, window, padding=padd, groups=channel)
    if mean_metric:
        return l1.mean()
    else:
        return l1

def ms_ssim_l1_loss(img1, img2, mean_metric = True, alpha=0.84):

    ms_ssim_map = msssim(img1, img2, mean_metric=False)
    l1_map = gauss_weighted_l1(img1, img2, mean_metric=False)
    loss_map = (1 - ms_ssim_map) * alpha + l1_map * (1 - alpha)
    if mean_metric:
        return loss_map.mean()
    else:
        return loss_map
def total_variation_regularization(images):
    width_var = nn.MSELoss(images[:,:-1,:,:] - images[:,1:,:,:])
    height_var = nn.MSELoss(images[:,:,:-1,:] - images[:,:,1:,:])
    TV_map = torch.add(width_var, height_var)
    return TV_map.mean()

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    def forward(self, img):
        return total_variation_regularization(img)

class MSSSIML1_Loss(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(MSSSIML1_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        # TODO: store window between calls if possible
        return ms_ssim_l1_loss(img1, img2)
