import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LRDataset(data.Dataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.LR_env = None  # environment for lmdb
        'CR'
        self.paths_WF = None
        self.WF_env = None  # environment for lmdb

        # read image list from lmdb or image files
        self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        assert self.paths_LR, 'Error: LR paths are empty.'
        'CR'
        self.WF_env, self.paths_WF = util.get_image_paths(opt['data_type'], opt['dataroot_WF'])
        assert self.paths_WF, 'Error: WF paths are empty.'

    def __getitem__(self, index):
        LR_path = None

        # get LR image
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(self.LR_env, LR_path)
        H, W, C = img_LR.shape

        'CR'
        # get WF image
        WF_path = self.paths_WF[index]
        img_WF = util.read_img(self.WF_env, WF_path)
        H_WF, W_WF, C_WF = img_WF.shape

        print('###########')
        print(img_LR.shape)
        print('###########')
        print(img_WF.shape)

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        'CR'
        # change color space if necessary
        if self.opt['color']:
            img_WF = util.channel_convert(C_WF, self.opt['color'], [img_WF])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_WF.shape[2] == 3:
            img_WF = img_WF[:, :, [2, 1, 0]]
        img_WF = torch.from_numpy(np.ascontiguousarray(np.transpose(img_WF, (2, 0, 1)))).float()

        print('###########')
        print(img_LR.shape)
        print('###########')
        print(img_WF.shape)
        img_LW = torch.cat((img_LR, img_WF), 0)
        return {'LR': img_LR, 'LR_path': LR_path, 'WF': img_WF, 'WF_path': WF_path, 'LW': img_LW}

    def __len__(self):
        return len(self.paths_LR)
