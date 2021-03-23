from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import cv2
import glob
import torch

class ISBI2012(Dataset):
    def __init__(self, path, n_channel=1, mode='train'):
        super(ISBI2012, self).__init__()

        self.n_channel = n_channel
        self.mode = mode
        self.images_path = glob.glob('%s/%s/images/*' % (path, mode))
        self.images_path.sort()

        self.transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2),
            A.Flip(),
            A.ElasticTransform(sigma=10, interpolation=cv2.INTER_CUBIC)
        ])

    def __getitem__(self, index):
        if self.n_channel == 1:
            img = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.imread(self.images_path[index]).astype(np.float32)
        img /= 255.
        if not self.mode == 'pred':
            label_path = self.images_path[index].replace('image', 'label', 2)
            if self.n_channel == 1:
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                label = np.expand_dims(label, axis=-1)
            else:
                label = cv2.imread(label_path).astype(np.float32)
            label /= 255.
        else:
            label = np.zeros(img.shape, dtype=np.float32)

        if self.mode == 'train':
            transformed = self.transforms(image=img, mask=label)
            img = transformed['image'].transpose((2, 0, 1))
            label = transformed['mask'].transpose((2, 0, 1))
        else:
            img = img.transpose((2, 0, 1))
            label = label.transpose((2, 0, 1))

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.images_path)