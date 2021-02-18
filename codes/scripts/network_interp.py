import os.path as osp
import sys
import torch

try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    import models.modules.RRDBNet_arch as RRDBNet_arch
except ImportError:
    pass

pretrained_net = torch.load('/home/rchenbe/rchenbe/BasicSR-netD12-gray_dual_input/experiments/Net_2input_1fps_training_ms_ssim_l1_lr3e-4_200500G_continue_lr1.5e-4_90600_continue_lr2e-4_140100_continue3e-4/models/82100_G.pth')
pretrained_net1 = torch.load('/home/rchenbe/rchenbe/BasicSR-netD12-gray_dual_input/experiments/Net_2input_1fps_training_ms_ssim_l1_lr3e-4_200500G_continue_lr1.5e-4_90600_continue_lr2e-4_140100_continue3e-4_continue_0.9P_0.1F_lr1e-4_continue_1P_0.1F_2.5e-4G_lr1e-4_continue_from_50600G_lr5e-5/models/108800_G.pth')
crt_model = RRDBNet_arch.RRDBNet(in_nc=2, out_nc=1, nf=64, nb=23)
crt_net = crt_model.state_dict()

for k, v in crt_net.items():
    if k in pretrained_net:
        if 'conv_last' not in k:
            crt_net[k] = 0.31 * pretrained_net[k] + 0.69 * pretrained_net1[k]
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

torch.save(crt_net, '/home/rchenbe/rchenbe/BasicSR-netD12-gray_dual_input/pretained_network\interp3169_2.pth')
