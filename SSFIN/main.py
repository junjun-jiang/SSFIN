from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from model import SpatialSpectralSRNet
from data import get_patch_training_set, get_test_set
from torch.nn import init
import skimage.measure
from torch.autograd import Variable
from psnr import MPSNR
import numpy as np
import math
import scipy.io as io
import os
import random
from torch.utils.tensorboard import SummaryWriter
# from thop import profile


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--ChDim', type=int, default=31, help='output channel number')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
parser.add_argument('--nEpochs', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0008, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
parser.add_argument('--mode', default=1, type=int, help='Train or Test.')
opt = parser.parse_args()

print(opt)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(opt.seed)

print('===> Loading datasets')
train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                                  pin_memory=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False,
                                 pin_memory=True)

print('===> Building model')
writer = SummaryWriter()

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

mkdir(opt.save_folder)
mkdir(opt.outputpath)

model = SpatialSpectralSRNet().cuda()
print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[5, 15, 30, 60], gamma=0.5)

if opt.nEpochs != 0:
    load_dict = torch.load(opt.save_folder + "_epoch_{}.pth".format(opt.nEpochs))
    opt.lr = load_dict['lr']
    epoch = load_dict['epoch']
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])

criterion = nn.L1Loss()

def train(epoch, optimizer):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        with torch.autograd.set_detect_anomaly(True):
            Y, X_1, X_2, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            optimizer.zero_grad()
            Y = Variable(Y).float()
            X_1 = Variable(X_1).float()
            X_2 = Variable(X_2).float()
            X = Variable(X).float()
            spa_X, spe_X, HX = model(Y)
            if epoch <= 10:
                alpha = opt.alpha
            elif epoch > 10 and epoch < 90:
                alpha = opt.alpha * math.cos(np.pi * (epoch - 10) / 160)
            else:
                alpha = 0

            loss = criterion(HX, X) + alpha * criterion(spe_X, X_2) + alpha * criterion(spa_X, X_1)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()


            if iteration%100==0:
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            Y, X = batch[0].cuda(), batch[1].cuda()
            Y = Variable(Y).float()
            X = Variable(X).float()
            spa_X, spe_X, HX = model(Y)
            X = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()
            spa_X = torch.squeeze(spa_X).permute(1, 2, 0).cpu().numpy()
            spe_X = torch.squeeze(spe_X).permute(1, 2, 0).cpu().numpy()
            psnr = MPSNR(HX, X)
            im_name = batch[2][0]
            print(im_name)
            (path, filename) = os.path.split(im_name)
            io.savemat(opt.outputpath + filename, {'HX': HX})
            io.savemat('spa/' + filename, {'HX': spa_X})
            io.savemat('spe/' + filename, {'HX': spe_X})
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)


def checkpoint(epoch):

    model_out_path = opt.save_folder + "_epoch_{}.pth".format(epoch)
    if epoch % 5 == 0:
        save_dict = dict(
            lr=optimizer.state_dict()['param_groups'][0]['lr'],
            param=model.state_dict(),
            adam=optimizer.state_dict(),
            epoch=epoch
        )
        torch.save(save_dict, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))


if opt.mode == 1:
    for epoch in range(opt.nEpochs + 1, 101):
        avg_loss = train(epoch, optimizer)
        checkpoint(epoch)
        scheduler.step()
else:
    test()
