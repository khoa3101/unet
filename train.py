import torch
import random
import numpy as np
import os
import argparse
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
from dataset import ISBI2012, KiTS2019
from loss import FocalTverskyLoss, DiceScore, DiceLoss, DiceCELoss
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
parser.add_argument('-g', '--group', default=False, type=bool, help='Use group convolutional layer or not')
parser.add_argument('-d', '--dataset', default='ISBI2012', choices=['ISBI2012', 'KiTS2019'], type=str, help='Dataset to train on')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
LR = args.learning_rate
EPOCH = args.epoch
NUM_CLASS = args.n_class
DATA_PATH = args.path
NUM_CHANNEL = args.n_channel
BEST = 0.0
SEED = args.seed
GROUP = args.group
DATASET = args.dataset
EXT = 'gconv' if GROUP else ''
FUNCTION = {
    'ISBI2012': ISBI2012,
    'KiTS2019': KiTS2019
}

def create_folders(path=''):
    if not os.path.isdir(os.path.join(path, 'checkpoints')):
        os.mkdir(os.path.join(path, 'checkpoints'))
    if not os.path.isdir(os.path.join(path, 'results')):
        os.mkdir(os.path.join(path, 'results'))

def train(epoch, model, train_loader, val_loader, optimizer, scheduler, loss, score, score_sub):
    global COUNT, BEST
    train_bar = tqdm(train_loader)
    train_loss = 0.0
    train_score = 0.0
    train_score_sub = 0.0
    train_batch = 0
    model.train()
    
    for image, label in train_bar:
        print(image.unique())
        train_batch += 1

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            
        optimizer.zero_grad()

        pred = model(image)
        # print(pred.unique())
        _loss = loss(pred, label.argmax(1).long())
        _score = score(pred, label)
        _score_sub = score_sub(pred, label)

        _loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += _loss.item()
        train_score += _score.item()
        train_score_sub += _score_sub.item()

        train_bar.set_description(desc='[%d/%d] Combine Loss: %.4f, Dice Score: %.4f, Dice Score sub: %.4f' % (
            epoch+1, EPOCH, train_loss/train_batch, train_score/train_batch, train_score_sub/train_batch
        ))

    val_score, val_score_sub , _ = val(model, val_loader, score, score_sub)

    if val_score_sub > BEST:
        BEST = val_score_sub
        torch.save(model.state_dict(), 'checkpoints/best_%s%s.pth' % (DATASET, '_' + EXT if EXT else ''))

    print('Best val Dice Score: %.4f' % BEST)

    return train_loss/train_batch, train_score/train_batch, train_score_sub/train_batch, val_score, val_score_sub

def val(model, val_loader, score, score_sub):
    val_bar = tqdm(val_loader)
    val_score = 0.0
    val_score_sub = 0.0
    val_batch = 0
    preds = []
    model.eval()

    with torch.no_grad():
        for image, label in val_bar:
            val_batch += 1
            
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            pred = model(image)
            _score = score(pred, label)
            _score_sub = score_sub(pred, label)
            val_score += _score.item()
            val_score_sub += _score_sub.item()

            if torch.cuda.is_available():
                pred = pred.detach().cpu()
            preds.append(pred.numpy().transpose((0, 2, 3, 1))[0])

            val_bar.set_description(desc='Dice Score: %.4f, Dice Score sub: %.4f' % (
                val_score/val_batch, val_score_sub/val_batch
            ))
    
    return val_score/val_batch, val_score_sub/val_batch, preds


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
    if DATASET == 'ISBI2012':
        train_data = FUNCTION[DATASET](DATA_PATH, n_channel=NUM_CHANNEL, mode='train')
        val_data = FUCTION[DATASET](DATA_PATH, n_channel=NUM_CHANNEL, mode='test')
    else:
        data = KiTS2019(DATA_PATH, n_channel=NUM_CHANNEL, mode='train')
        n_val = int(len(data) * 0.1) + 1
        n_train = len(data) - n_val
        train_data, val_data = random_split(data, [n_train, n_val])
    train_loader = DataLoader(train_data, num_workers=8, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, num_workers=8, batch_size=2*BATCH_SIZE, shuffle=False)

    # Prepare loss and model
    axis = tuple(i for i in range(1, NUM_CLASS)) if NUM_CLASS > 1 else None
    print(axis)
    # loss = FocalTverskyLoss()
    loss = nn.CrossEntropyLoss()#DiceCELoss() if NUM_CLASS == 1 else DiceCELoss(binary=False, axis=axis)
    score = DiceScore(threshold=0.5, axis=axis[:-1])
    score_sub = DiceScore(axis=NUM_CLASS-1, threshold=0.5)
    if GROUP:
        model = UNetGConv(n_channels=NUM_CHANNEL, n_classes=NUM_CLASS)
    else:
        model = UNet(n_channels=NUM_CHANNEL, n_classes=NUM_CLASS)
    print('Trainable params: %s' % model.total_trainable_params())
    print('Total params: %s' % model.total_params())
    if torch.cuda.is_available():
        loss.cuda()
        model.cuda()

    # Prepare optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-8)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if NUM_CLASS > 1 else 'max', patience=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.00001)
    print('Done\n')

    result = {
        'train_loss': [],
        'train_score': [],
        'train_score_sub': [],
        'val_score': [],
        'val_score_sub': []
    }
    print('Training...')
    for epoch in range(EPOCH):
        train_loss, train_score, train_score_sub, val_score, val_score_sub = train(
            epoch, model, train_loader, val_loader, optimizer, scheduler, loss, score, score_sub
        )
        result['train_loss'].append(train_loss)
        result['train_score'].append(train_score)
        result['train_score_sub'].append(train_score_sub)
        result['val_score'].append(val_score)
        result['val_score_sub'].append(val_score_sub)
        
    data_frame = pd.DataFrame(
        data={
            'Train loss': result['train_loss'], 'Train dice': result['train_score'], 'Train dice sub': result['train_score_sub'],
            'Val dice': result['val_score'], 'Val dice sub': result['val_score_sub']
        },
        index=range(1, epoch+2)
    )
    data_frame.to_csv('results/training_%s%s.csv' % (DATASET, '_' + EXT if EXT else ''), index_label='Epoch') 
    print('Done')
