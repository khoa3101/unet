from albumentations.augmentations.transforms import HorizontalFlip, RandomBrightness, RandomContrast, RandomGamma
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import cv2
import glob
import torch
from albumentations.pytorch.transforms import ToTensorV2

class ISBI2012(Dataset):
    def __init__(self, path, n_channel=1, mode='train'):
        super(ISBI2012, self).__init__()

        self.n_channel = n_channel
        self.mode = mode
        self.images_path = glob.glob('%s/%s/images/*' % (path, mode))
        self.images_path.sort()

        self.train_transforms = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2),
                A.Flip(),
                A.GridDistortion(),
                A.ElasticTransform(sigma=10, interpolation=cv2.INTER_CUBIC),
                ToTensorV2()
            ]
        )

        self.val_transforms = A.Compose([
            ToTensorV2()
        ])

    def __getitem__(self, index):
        image_path = self.images_path[index]
        img = cv2.imread(image_path, 1)
        label_path = image_path.replace('image', 'label', 2)
        label = cv2.imread(label_path, 0)

        if self.mode == 'train':
            transformed = self.train_transforms(image=img, mask=label)
        else:
            transformed = self.val_transforms(image=img, mask=label)
        
        img = transformed['image']
        label = transformed['mask']

        img, label = torch.div(img, 255.), torch.div(label, 255.)

        return img.float(), label.float().unsqueeze(0)

    def __len__(self):
        return len(self.images_path)


class SIIM(Dataset):
    def __init__(self, path, size=256, n_channel=1, mode='train'):
        super(SIIM, self).__init__()

        self.size = size
        self.n_channel = n_channel
        self.mode = mode

        self.images_path = glob.glob('%s/%s/images/*' % (path, mode))
        self.images_path.sort()
        
        self.train_transforms = A.Compose([
            A.HorizontalFlip(),
            A.OneOf([
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness()
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5)
            ], p=0.3),
            A.RandomResizedCrop(min_max_height=(176, 256), height=size, width=size, p=0.25),
            ToTensorV2()
        ])

        self.val_transforms = A.Compose([
            ToTensorV2()
        ])

    def __getitem__(self, index):
        image_path = self.images_path[index]
        img = cv2.imread(image_path, 1)
        img = cv2.resize(img, (self.size, self.size))
        label_path = image_path.replace('image', 'label', 2)
        try:
            label = cv2.imread(label_path, 0)
        except:
            label = np.zeros((img.shape, img.shape), dtype=np.uint8)
        label = cv2.resize(label, (self.size, self.size))

        if self.mode == 'train':
            transformed = self.train_transforms(image=img, mask=label)
        else:
            transformed = self.val_transforms(image=img, mask=label)
        
        img = transformed['image']
        label = transformed['mask']

        img, label = torch.div(img, 255.), torch.div(label, 255.)

        return img.float(), label.float().unsqueeze(1)
    
    def __len__(self):
        return len(self.images_path)