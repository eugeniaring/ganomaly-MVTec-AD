from operator import truediv
import os
# import tarfile
from PIL import Image
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]


class MVTecDataset(Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 class_name='bottle',
                 is_train=True,
                 resize=256,
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize, Image.NEAREST),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

def import_data(params,is_train=True):
    
    if is_train == True:
        dataset = MVTecDataset(params['data_path'], class_name=params['obj'], is_train=is_train, resize=params['img_size'])
        img_nums = len(dataset)
        valid_num = int(img_nums * params['validation_ratio'])
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(dataset, [train_num, valid_num])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
        return dataset, train_loader, val_loader
    else:
        dataset = MVTecDataset(params['data_path'], class_name=params['obj'], is_train=is_train, resize=params['img_size'])
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        return dataset, test_loader


    