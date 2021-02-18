import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTLoss(nn.Module):
    '''Compute the Fourier spectrum of the image to compare frequencies
    It is done by patch so the information is not totally global information
    In this implementation the windows are of size 32x32 (out of a 256x256 image or RandomCrop)'''


    def __init__(self):
        super(FFTLoss, self).__init__()
        self.eps = 1e-7


    def forward(self, x):
        # Extract the FFT using torch algorithm
        vF = torch.rfft(x, 2, onesided=False)
        # get real part
        vR = vF[:,:,33:33+192-1,33:33+192-1, 0]
        # get the imaginary part
        vI = vF[:,:,33:33+192-1,33:33+192-1, 1]
        # Get spectrum by computing the elemcent wise complex modulus
        out = torch.add(torch.pow(vR, 2), torch.pow(vI, 2))
        out = torch.sqrt(out + self.eps)

        return out
