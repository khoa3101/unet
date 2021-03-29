import cv2
import os
import glob
import argparse
import numpy as np
from skimage import measure
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Preprocessing label to get weight map')
parser.add_argument('-p', '--path', default='ISBI 2012', type=str, help='Path to data folder')
opt = parser.parse_args()

DATA_PATH = opt.path

def create_folder():
    if not os.path.isdir('%s/train/weights' % DATA_PATH):
        os.mkdir('%s/train/weights' % DATA_PATH)
    if not os.path.isdir('%s/test/weights' % DATA_PATH):
        os.mkdir('%s/test/weights' % DATA_PATH)

def weight(path):
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gt = 1 * (gt > 0)

    # 1 - Calculate the pixel frequency of cells and background
    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())

    # 2 - Normalization
    c_weights /= c_weights.max()

    # 3 - Get class_weight map (cw_map)
    cw_map = np.where(gt==0, c_weights[0], c_weights[1])

    # 4 - Connected domain analysis
    cells = measure.label(gt, connectivity=2)

    # 5 - Calculate the distance weight map (dw_map)
    w0 = 10
    sigma = 5
    dw_map = np.zeros_like(gt)
    maps = np.zeros((gt.shape[0], gt.shape[1], cells.max()))
    if cells.max() >= 2:
        for i in range(1, cells.max() + 1):
            maps[:,:,i-1] =  cv2.distanceTransform(1- (cells == i ).astype(np.uint8), cv2.DIST_L2, 3)
        maps = np.sort(maps, axis = 2)
        d1 = maps[:,:,0]
        d2 = maps[:,:,1]
        dis = ((d1 + d2)**2) / (2 * sigma * sigma)
        dw_map = w0*np.exp(-dis) * (cells == 0)
    
    return dw_map + cw_map

def save_map(weight_map, path):
    path = path.replace('label', 'weight', 2)
    path = path.replace('jpg', 'npy')
    with open(path, 'wb') as f:
        np.save(f, weight_map)


if __name__ == '__main__':
    print('Creating folders..')
    create_folder()
    print('Done\n')

    for mode in ['train', 'test']:
        path = glob.glob('%s/%s/labels/*' % (DATA_PATH, mode))
        for p in tqdm(path):
            weight_map = weight(p)
            save_map(weight_map, p)  
    print('Done')

    # with open('ISBI 2012/test/weights/train-weights00.npy', 'rb') as f:
    #     a = np.load(f)
    # print(a) 