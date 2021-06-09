import os
import sys
import time
import shutil
import argparse
import cv2 

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from dataset import *
from utils import *
from model import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='device name, cuda:0')
    parser.add_argument('--experiment', required=True,
                        help='prefix of outputs, e.g., experiment_best_model.pth will be saved to ckpt/')
    parser.add_argument('--featureExt', type=str, required=True, help='either: resnet50, mobilenetv2, vgg16, densenet, squeezenet')
    parser.add_argument('--base_Dir', type=str, required=True,
                        help='base_Dir/train, base_Dir/val will be used')
    parser.add_argument('--gt_Dir', type=str, required=True,help='directory for groundtruth NLIs')
    parser.add_argument('--ckpt_Dir',type=str, required=True,help='directory for saving model weights')
    parser.add_argument('--numEpoch', type=int, default=30)
    args = parser.parse_args()
    return args

args = parse_args()

if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)


basedir = args.base_Dir
train_dir = os.path.join(basedir, 'train')
val_dir = os.path.join(basedir, 'val')
gt_dir = args.gt_Dir

train_dataset = SICEPart1(train_dir, gt_dir, transform=transforms.ToTensor())
trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
val_dataset = SICEPart1(val_dir, gt_dir, transform=transforms.ToTensor())
valloader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

loaders = {'train': trainloader, 'val': valloader}

def train(loaders, model, optimizer, scheduler, epoch, num_epochs):
    model = model.train()
    print(f'--- Epoch {epoch} ---')
    total = []
    for i, sample in enumerate(loaders['train']):
        optimizer.zero_grad()
        img_batch = sample['lowImg'].to(device)
        high_batch = sample['highImg'].to(device)
        filterHF = model(img_batch)  
        filterHF = filterHF.reshape (img_batch.size(0),2)
        print(filterHF)
        enhanced_batch = refine_image(img_batch, filterHF,device)
        l = (filterHF [:,0]-filterHF[:,1]).mean()
        ms_ssim = mssim (enhanced_batch,high_batch)
        loss = ms_ssim + 0.08*l
        total.append(loss)
        loss.backward()
        optimizer.step()
    print ("\n train loss: " + str(sum(total)/len(total)))
    total = []

    model = model.eval()
    with torch.no_grad():
        for i, sample in enumerate(loaders['val']):
            img_batch = sample['lowImg'].to(device)
            high_batch = sample['highImg'].to(device)
            filterHF = model(img_batch)
            filterHF = filterHF.reshape (img_batch.size(0),2)
            enhanced_batch = refine_image(img_batch, filterHF,device)
            l = (filterHF [:,0]-filterHF[:,1]).mean()
            ms_ssim = mssim (enhanced_batch,high_batch)
            loss = ms_ssim + 0.08*l
            total.append(loss.item())
    print ("\n val loss: " + str(sum(total)/len(total)))
    val_loss = np.mean(total)
    scheduler.step(val_loss)
    return val_loss


hp = dict(lr=1e-4, wd=1e-4, lr_decay_factor=0.99)
model = switch_model(args.featureExt)
model.to(device)
grouped_params = group_params(model)  
optimizer = optim.Adam(grouped_params, lr=hp['lr'], weight_decay=hp['wd'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.numEpoch // 10, mode='min', factor=hp['lr_decay_factor'], threshold=3e-4)
experiment = args.experiment
ckpt_dir = args.ckpt_Dir
loss_history, best_loss = [], float('inf')
num_epochs = args.numEpoch

print(f'[START TRAINING JOB] -{experiment}')

for epoch in range(num_epochs):
    val_loss = train(loaders, model, optimizer, scheduler, epoch, num_epochs)
    is_best = val_loss < best_loss
    best_loss = min(val_loss, best_loss)
    save_ckpt({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'HyperParam': hp,
        'val_loss': val_loss,
        'loss_history': loss_history,
        'model_src': open('./model.py', 'rt').read(),
        'train_src': open('./train.py', 'rt').read()
    }, is_best, experiment, epoch, ckpt_dir)


