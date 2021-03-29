from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import cv2
import glob
import torch
from albumentations.pytorch.transforms import ToTensor, ToTensorV2

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
                # A.RandomBrightnessContrast(),
                A.GridDistortion(),
                # A.ElasticTransform(sigma=10, interpolation=cv2.INTER_CUBIC),
                ToTensorV2()
                # A.GaussNoise()
            ],
            # additional_targets={'weight_map': 'image'}
        )

        self.val_transforms = A.Compose(
            [
                ToTensorV2()
            ]
        )

    # def load_image(self, path):
    #     if self.n_channel == 1:
    #         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #     else:
    #         img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    #     return img

    def __getitem__(self, index):
        image_path = self.images_path[index]
        img = cv2.imread(image_path, 0)
        label_path = image_path.replace('image', 'label', 2)
        label = cv2.imread(label_path, 0)

        # weight_path = self.images_path[index].replace('image', 'weight', 2)
        # weight_path = weight_path.replace('jpg', 'npy')
        # with open(weight_path, 'rb') as f:
        #     weight_map = np.load(f)

        if self.mode == 'train':
            transformed = self.train_transforms(image=img, mask=label) # , weight_map=weight_map)
            # weight_map = transformed['weight_map']
        else:
            transformed = self.val_transforms(image=img, mask=label)
        
        img = transformed['image']
        label = transformed['mask']

        img, label = torch.div(img, 255.), torch.div(label, 255.)

        return img.float().unsqueeze(1), label.float().unsqueeze(1), torch.ones(img.shape) # torch.from_numpy(weight_map)

    def __len__(self):
        return len(self.images_path)