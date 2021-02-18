import math
import torchvision.utils
from data import create_dataloader, create_dataset
from utils import util

opt = {}

opt['name'] = 'test_img_id'
opt['dataroot_GT'] = '/home/rchenbe/rchenbe/BasicSR-netD12-gray_dual_input/data_samples/microtubule/train/HR'
opt['dataroot_LQ'] = '/home/rchenbe/rchenbe/BasicSR-netD12-gray_dual_input/data_samples/microtubule/train/SRRF'
opt['dataroot_WF'] = '/home/rchenbe/rchenbe/BasicSR-netD12-gray_dual_input/data_samples/microtubule/train/LR'
opt['mode'] = 'LQGT'
opt['phase'] = 'train'  # 'train' | 'val'
opt['use_shuffle'] = True
opt['n_workers'] = 8
opt['batch_size'] = 16
opt['GT_size'] = 128
opt['scale'] = 4
opt['use_flip'] = True
opt['use_rot'] = True
#opt['color'] = 'RGB'
opt['data_type'] = 'img'  # img | lmdb
opt['dist'] = False
opt['gpu_ids'] = [0]

util.mkdir('tmp')
train_set = create_dataset(opt)
train_loader = create_dataloader(train_set, opt, opt, None)
nrow = int(math.sqrt(opt['batch_size']))
if opt['phase'] == 'train':
    padding = 2
else:
    padding = 0

for i, data in enumerate(train_loader):
    if i > 3:
        break
    print(i)
    LQ = data['LQ']
    WF = data['WF']
    GT = data['GT']
    torchvision.utils.save_image(LQ, 'tmp/LQ_{:03d}.png'.format(i), nrow=nrow, padding=padding,
                                 normalize=False)
    torchvision.utils.save_image(GT, 'tmp/GT_{:03d}.png'.format(i), nrow=nrow, padding=padding,
                                 normalize=False)
    torchvision.utils.save_image(WF, 'tmp/WF_{:03d}.png'.format(i), nrow=nrow, padding=padding,
                                 normalize=False)
