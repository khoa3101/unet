import torch 
import torch.nn.functional as F
import cv2
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm 
from data import ISBI2012
from loss import DiceScore
from model import UNet
from model1 import DcUnet


parser = argparse.ArgumentParser(description='Unet')
parser.add_argument('-p', '--path', default='ISBI2012', type=str, help='Path to data folder')
parser.add_argument('-t', '--threshold', default=0.5, type=float, help='Threshold of confidence')
parser.add_argument('-ckpt', '--checkpoint', default='', type=str, help='Path to the checkpoint of model')
parser.add_argument('-n', '--n_class', default=1, type=int, help='Number of class to segmentation')
parser.add_argument('-c', '--n_channel', default=3, type=int, help='Number of channels')
opt = parser.parse_args()

NUM_CLASS = opt.n_class
NUM_CHANNEL = opt.n_channel
THRESHOLD = opt.threshold
DATA_PATH = opt.path
PATH = 'checkpoints/best_%s.pth' % opt.path if not opt.checkpoint else opt.checkpoint
RESULT_PATH = '%s/test/preds' % opt.path

def save_preds(preds, result_path):
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    for idx, pred in enumerate(preds):
        pred = (pred * 255).astype(np.uint8)
        cv2.imwrite('%s/%s.tif' % (result_path, str(idx).zfill(2)), pred)

def test(model, test_loader, score, threshold=0.5):
    # test_bar = tqdm(testt_loader)
    test_result = 0.0
    test_batch = 0
    preds = []
    model.eval()

    for image, label, _ in test_loader:
        test_batch += 1
        
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        pred = model(image)
        dice_score = score(pred, label, threshold)
        test_result += dice_score.item()

        pred = torch.sigmoid(pred)
        pred = (pred > THRESHOLD).float()
        if torch.cuda.is_available():
            pred = pred.detach().cpu()
        preds.append(pred.numpy().transpose((0, 2, 3, 1))[0])

        # test_bar.set_description(desc='Dice Score: %.4f' % (test_result/test_batch))
    
    return test_result/test_batch, preds


if __name__ == '__main__':
    print('Preparing...')

    # Prepare data
    test_data = ISBI2012(DATA_PATH, n_channel=NUM_CHANNEL, mode='test')
    test_loader = DataLoader(test_data, num_workers=2, pin_memory=True)

    # Prepare loss and model
    score = DiceScore()
    model = UNet(n_channels=NUM_CHANNEL, n_class=NUM_CLASS)
    # model = DcUnet(NUM_CHANNEL)
    model.load_state_dict(torch.load(PATH))
    if torch.cuda.is_available():
        score.cuda()
        model.cuda()
    print('Done\n')

    # Predicting and save images
    print('Predicting and saving...')
    result, preds = test(model, test_loader, score, THRESHOLD)
    print('Dice score: %.4f' % result)
    save_preds(preds, RESULT_PATH)
    print('Done')
