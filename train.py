import torch
import random
import numpy as np
import os
import argparse
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
from data import ISBI2012
from loss import FocalTverskyLoss, DiceScore, DiceLoss, DiceBCELoss
from model import UNet
from model_gconv import UNetGConv


parser = argparse.ArgumentParser(description='Unet')
parser.add_argument('-p', '--path', default='ISBI2012', type=str, help='Path to data folder')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate of optimzer')
parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch size of dataloader')
parser.add_argument('-e', '--epoch', default=5, type=int, help='Epoch to train model')
parser.add_argument('-n', '--n_class', default=1, type=int, help='Number of class to segmentation')
parser.add_argument('-c', '--n_channel', default=3, type=int, help='Number of channels')
parser.add_argument('-s', '--seed', default=31, type=str, help='Random seed')
opt = parser.parse_args()

BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
EPOCH = opt.epoch
NUM_CLASS = opt.n_class
DATA_PATH = opt.path
NUM_CHANNEL = opt.n_channel
COUNT = 0
BEST = 0.0
SEED = opt.seed

def create_folders(path=''):
    if not os.path.isdir(os.path.join(path, 'checkpoints')):
        os.mkdir(os.path.join(path, 'checkpoints'))
    if not os.path.isdir(os.path.join(path, 'results')):
        os.mkdir(os.path.join(path, 'results'))

def train(epoch, model, train_loader, val_loader, optimizer, scheduler, scaler, loss1, score):
    global COUNT, BEST
    train_bar = tqdm(train_loader)
    train_result = 0.0
    train_score = 0.0
    train_batch = 0
    model.train()
    
    for image, label in train_bar:
        train_batch += BATCH_SIZE
        COUNT += 1

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(image)
            _loss = loss1(pred, label)
            _score = score(pred, label, 0.5)
        scaler.scale(_loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        train_result += _loss.item()
        train_score += _score.item()

        train_bar.set_description(desc='[%d/%d] Combine Loss: %.4f, Dice Score: %.4f' % (epoch+1, EPOCH, _loss.item(), _score.item()))

    val_result, _ = val(model, val_loader, score)
    print('Dice score: %.4f' % val_result)

    if val_result > BEST:
        BEST = val_result
        torch.save(model.state_dict(), 'checkpoints/best_%s.pth' % DATA_PATH)

    return train_result/train_batch

def val(model, val_loader, score, threshold=0.5):
    # val_bar = tqdm(val_loader)
    val_result = 0.0
    val_batch = 0
    preds = []
    model.eval()

    for image, label in val_loader:
        val_batch += 1
        
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        pred = model(image)
        dice_score = score(pred, label, threshold)
        val_result += dice_score.item()

        if torch.cuda.is_available():
            pred = pred.detach().cpu()
        preds.append(pred.numpy().transpose((0, 2, 3, 1))[0])

        # val_bar.set_description(desc='Dice Score: %.4f' % (val_result/val_batch))
    
    return val_result/val_batch, preds


if __name__ == '__main__':
    # Set seed
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Create folders for storing training results
    print('Creating folders...')
    create_folders()
    print('Done\n')

    print('Preparing...')
    # Prepare data
    train_data = ISBI2012(DATA_PATH, n_channel=NUM_CHANNEL, mode='train')
    val_data = ISBI2012(DATA_PATH, n_channel=NUM_CHANNEL, mode='test')
    # n_val = int(len(dataset) * 0.1) + 1
    # n_train = len(dataset) - n_val
    # train_data, val_data = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_data, num_workers=2, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, num_workers=2, pin_memory=True, batch_size=1, shuffle=False)

    # Prepare loss and model
    # loss1 = FocalTverskyLoss()
    loss1 = DiceBCELoss()
    score = DiceScore()
    # model = UNet(n_channels=NUM_CHANNEL, n_classes=NUM_CLASS)
    model = UNetGConv(n_channels=NUM_CHANNEL, n_classes=NUM_CLASS)
    if torch.cuda.is_available():
        loss1.cuda()
        model.cuda()

    # Prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-8)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if NUM_CLASS > 1 else 'max', patience=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.00001)
    print('Done\n')

    # Prepare scaler
    scaler = torch.cuda.amp.GradScaler()

    result = []
    print('Training...')
    for epoch in range(EPOCH):
        train_result = train(epoch, model, train_loader, val_loader, optimizer, scheduler, scaler, loss1, score)
        result.append(train_result)
        
    data_frame = pd.DataFrame(
        data={'Dice loss': result},
        index=range(1, epoch+2)
    )
    data_frame.to_csv('results/training_%s.csv' % DATA_PATH, index_label='Epoch') 
    print('Done')
