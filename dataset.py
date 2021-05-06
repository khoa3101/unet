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

        self.train_transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2),
            A.Flip(),
            A.GridDistortion(),
            A.ElasticTransform(sigma=10, interpolation=cv2.INTER_CUBIC),
            ToTensorV2()
        ])

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


class KiTS2019(Dataset):
    def __init__(self, path, n_channel=1, mode='train'):
        super(KiTS2019, self).__init__()

        self.n_channel = n_channel,
        self.mode = mode
        self.images_path = glob.glob('%s/%s/images/*' % (path, mode))

        self.transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2),
            A.Flip(),
            A.GridDistortion(),
            A.ElasticTransform(sigma=10, interpolation=cv2.INTER_CUBIC),
        ])

    def __getitem__(self, index):
        image_path = self.images_path[index]
        img = np.load(image_path)
        label_path = image_path.replace('image', 'label')
        label = np.load(label_path)
        img = (img + 80.)/380.
        
        # if self.mode == 'train':
        #     transformed = self.transforms(image=img, mask=label)
        #     img = transformed['image']
        #     label = transformed['mask']
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        label = torch.stack((label==0, label==1, label==2), dim=0)

        if self.n_channel == 1:
            img = img.unsqueeze(0)
        else:
            img = torch.stack((img, img, img), dim=0)
        return img.float(), label.float()

    def __len__(self):
        return len(self.images_path)