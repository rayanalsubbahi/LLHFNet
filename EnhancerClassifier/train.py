import os
import sys
import time
import shutil
import argparse
import cv2 

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from utils import *
from model import *

import tfrecord

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='device name, cuda:0')
    parser.add_argument('--experiment', required=True,
                        help='prefix of outputs, e.g., experiment_best_model.pth will be saved to ckpt/')
    parser.add_argument('--base_Dir', type=str, required=True,
                        help='base_Dir/trainData.tfrecord, base_Dir/valData.tfrecord will be used')
    parser.add_argument('--numEpoch', type=int, default=30)
    parser.add_argument('--ckpt_Dir',type=str, required=True,help='directory for saving model weights')
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
train_dir = os.path.join(basedir, 'trainData.tfrecord')
val_dir = os.path.join(basedir, 'valData.tfrecord')

transform=transforms.ToTensor()

def decode_image(features):
    features["lowImage"] = cv2.imdecode(features ["lowImage"], -1)
    features["lowImage"] = transform (features["lowImage"])
    features["highImage"] = cv2.imdecode(features ["highImage"], -1)
    features["highImage"] = transform (features["highImage"])
    features['name'] = "".join([chr(value) for value in features['name']])
    return features

description = {
      "name": ("byte"), 
      "lowImage": ("byte"),
      "highImage": ("byte"),
      "label": ("int"),
}

train_dataset = tfrecord.torch.TFRecordDataset(train_dir,
                                         index_path=None,
                                         description=description,
                                         transform=decode_image,
                                         shuffle_queue_size = 8500)
trainloader = DataLoader(train_dataset, batch_size=8)

val_dataset = tfrecord.torch.TFRecordDataset(val_dir,
                                         index_path=None,
                                         description=description,
                                         transform=decode_image)
valloader = DataLoader(val_dataset, batch_size=1)
loaders = {'train': trainloader, 'val': valloader}

def train(loaders, model, optimizer, criterion, scheduler, epoch, num_epochs):
    model = model.train()
    print(f'--- Epoch {epoch} ---')
    total = []   
    for batch in loaders ['train']:
        optimizer.zero_grad()
        #input data
        img_batch = batch['lowImage'].to(device)
        high_batch = batch ['highImage'].to(device)
        labels = batch ['label'].reshape (1,batch ['label'].size(0))[0].long().to(device) 
        #model output
        outputs2, outputs1, enhanced_batch, filterHF = model (img_batch)

        #Enhacement loss
        l = (filterHF [:,0]-filterHF[:,1]).mean()
        ms_ssim = mssim (enhanced_batch,high_batch)
        lossEnh = ms_ssim + 0.08*l

        #Classification loss
        lossCls2 = criterion (outputs2,labels)
        lossCls1 = criterion (outputs1,labels)

        #total loss
        e = 1
        c = 0.1 
        if (epoch >= 1):
           e = 1
           c = 1
        loss = e*lossEnh + c*lossCls2 + c*lossCls1
        total.append(loss)
        loss.backward()
        optimizer.step()

    print ("\n train loss: " + str(sum(total)/len(total)))
    total = []

    model = model.eval()

    with torch.no_grad():
        for batch in loaders ['val']:
            #input data
            img_batch = batch['lowImage'].to(device)
            high_batch = batch ['highImage'].to(device)
            labels = batch ['label'].reshape (1,batch ['label'].size(0))[0].long().to(device) 
            #model output
            outputs2,outputs1, enhanced_batch, filterHF = model (img_batch)

            #Enhacement loss
            l = (filterHF [:,0]-filterHF[:,1]).mean()
            ms_ssim = mssim (enhanced_batch,high_batch)
            lossEnh = ms_ssim + 0.08*l

            #Classification loss
            lossCls2 = criterion (output2,labels)
            lossCls1 = criterion (outputs1,labels)
            #total loss
            e = 1
            c = 0.1
            if (epoch >= 1):
                e = 1
                c = 1
            loss = e*lossEnh + c*lossCls2 + c*lossCls1
            total.append (loss.item()) 
    print ("\n val loss: " + str(sum(total)/len(total)))
    val_loss = np.mean(total)
    scheduler.step(val_loss)
    return val_loss
    

hp = dict(lr=1e-5, wd=1e-4, lr_decay_factor=0.99)
model = EnhCls()
model.to(device)
grouped_params = group_params(model) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(grouped_params, lr=hp['lr'], weight_decay=hp['wd'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.numEpoch // 10, mode='min', factor=hp['lr_decay_factor'], threshold=3e-4)

experiment = args.experiment
ckpt_dir = args.ckpt_Dir
loss_history, best_loss = [], 0.0
num_epochs = args.numEpoch

print(f'[START TRAINING JOB] -{experiment}')

for epoch in range(num_epochs):
    val_loss = train(loaders, model, optimizer, criterion, scheduler, epoch, num_epochs)
    is_best = val_loss > best_loss
    best_loss = max(val_loss, best_loss)
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
