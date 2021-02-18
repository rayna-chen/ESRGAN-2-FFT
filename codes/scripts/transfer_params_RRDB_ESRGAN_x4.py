import os.path as osp
import sys
import torch

try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    import models.modules.RRDBNet_arch as RRDBNet_arch
except ImportError:
    pass

pretrained_net = torch.load('D:\program\BasicSR-netD12-gray_dual_input\pretained_network\RRDB_ESRGAN_x4.pth')
crt_model = RRDBNet_arch.RRDBNet(in_nc=2, out_nc=1, nf=64, nb=23)
crt_net = crt_model.state_dict()

for k, v in crt_net.items():
    if k in pretrained_net and 'conv_first' not in k:
        if 'conv_last' not in k:
            crt_net[k] = pretrained_net[k]
            print('replace ... ', k)



# # in: C3 -> C2
# crt_net['conv_first.weight'][:, 0:1, :, :] = pretrained_net['conv_first.weight'][:, 0:1, :, :] / 2
# crt_net['conv_first.weight'][:, 1:2, :, :] = pretrained_net['conv_first.weight'][:, 0:1, :, :] / 2
# crt_net['conv_first.bias'] = pretrained_net['conv_first.bias'] / 2
# crt_net['conv_first.bias'] = pretrained_net['conv_first.bias'] / 2
#
# #out: C3 -> C1
# crt_net['conv_last.weight'][:, :, :, :] = pretrained_net['conv_last.weight'][0:1, :, :, :]
# crt_net['conv_last.bias'] = pretrained_net['conv_last.bias'][0:1]

torch.save(crt_net, 'D:\program\BasicSR-netD12-gray_dual_input\pretained_network\RRDB_ESRGAN_x4_C2O1.pth')