from random import paretovariate
import torch
import os
import argparse
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader#, random_split
from tqdm import tqdm 
from data import ISBI2012
from loss import DiceLoss
from model import UNet


parser = argparse.ArgumentParser(description='Unet')
parser.add_argument('-p', '--path', default='ISBI 2012', type=str, help='Path to data folder')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate of optimzer')
parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch size of dataloader')
parser.add_argument('-e', '--epoch', default=5, type=int, help='Epoch to train model')
parser.add_argument('-n', '--n_class', default=1, type=int, help='Number of class to segmentation')
parser.add_argument('-c', '--n_channel', default=1, type=int, help='Number of channels')
opt = parser.parse_args()

BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
EPOCH = opt.epoch
NUM_CLASS = opt.n_class
DATA_PATH = opt.path
NUM_CHANNEL = opt.n_channel


def create_folders(path=''):
    if not os.path.isdir(os.path.join(path, 'checkpoints')):
        os.mkdir(os.path.join(path, 'checkpoints'))
    if not os.path.isdir(os.path.join(path, 'results')):
        os.mkdir(os.path.join(path, 'results'))

def train(epoch, model, train_loader, optimizer, loss, score):
    train_bar = tqdm(train_loader)
    train_result = 0.0
    train_score = 0.0
    train_batch = 0
    model.train()
    
    for image, label in train_bar:
        train_batch += BATCH_SIZE
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            
        pred = model(image)

        _loss = loss(pred, label)
        sub_pred = torch.sigmoid(pred)
        sub_pred = (sub_pred > 0.5).float()
        _score = score(sub_pred, label)

        optimizer.zero_grad()
        _loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

        train_result += _loss.item()
        train_score += _score.item()

        train_bar.set_description(desc='[%d/%d] BCE Loss: %.4f, Dice Score: %.4f' % (epoch+1, EPOCH, _loss.item(), _score.item()))

        # if train_batch == 10:
        #     scheduler.step()

    return train_result/train_batch

def val():
    pass


if __name__ == '__main__':
    # Create folders for storing training results
    print('Creating folders...')
    create_folders()
    print('Done\n')

    print('Preparing...')
    # Prepare data
    train_data = ISBI2012(DATA_PATH, n_channel=NUM_CHANNEL, mode='train')
    train_loader = DataLoader(train_data, num_workers=8, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True)

    # Prepare loss and model
    loss = nn.BCEWithLogitsLoss() # DiceLoss()
    score = DiceLoss()
    model = UNet(n_channels=NUM_CHANNEL, n_class=NUM_CLASS)
    if torch.cuda.is_available():
        loss.cuda()
        model.cuda()

    # Prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if NUM_CLASS > 1 else 'max', patience=2)
    print('Done\n')

    result = []
    best_result = 1.0e6
    print('Training...')
    for epoch in range(EPOCH):
        # print('Epoch %d' % epoch)

        train_result = train(epoch, model, train_loader, optimizer, loss, score)

        result.append(train_result)
        if train_result < best_result:
            best_result = train_result
            torch.save(model.state_dict(), 'checkpoints/best_%s.pth' % DATA_PATH)

        data_frame = pd.DataFrame(
            data={'Dice loss': result},
            index=range(1, epoch+2)
        )
        data_frame.to_csv('results/training_%s.csv' % DATA_PATH, index_label='Epoch') 
    print('Done')
